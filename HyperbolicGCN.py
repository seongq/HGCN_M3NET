import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from mathutils import cosh, sinh, arcosh
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
import math
import torch.nn.init as init
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from typing import Optional, Tuple, Union
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
class hyperbolicGCN(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels,dropout_rate = 0.5,curvature=0.1):
        super(hyperbolicGCN, self).__init__()
        self.num_layers = num_layers
        self.base_curvature = curvature
        self.curvatures = nn.ParameterList([nn.Parameter(torch.Tensor([curvature])).cuda() for _ in range(num_layers+1)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
       
        hgc_layers = []
        for i in range(self.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim = in_channels
            out_dim = out_channels
            hgc_layers.append(hyperbolicGCNblock(in_dim, out_dim, c_in, c_out, self.dropout_rate))
        self.hgc_layers = nn.ModuleList(hgc_layers)
        
    def encoder(self, x, edge_index):
        
        o = torch.zeros_like(x)
        x = torch.cat([o[:,0:1],x], dim=1)
        curvature_0 = F.softplus(self.curvatures[0])
        x_tan = proj_tan0(x, curvature_0)
        x_hyp = expmap0(x_tan, curvature_0)
        x_hyp = proj(x_hyp, curvature_0)
        
        return x_hyp
        
    def decoder(self, x, edge_index):
        curvature_last = F.softplus(self.curvatures[-1])
        h = proj_tan0(logmap0(x,curvature_last), curvature_last)
        return h
        
    def forward(self,x, edge_index):
        encoded = self.encoder(x,edge_index)
        
        
        
        
        for layer in self.hgc_layers:
            
            encoded = layer(encoded, edge_index)
            
            
        last_hyp_rep = encoded
        decoded = self.decoder(last_hyp_rep, edge_index)
        return decoded
        
class hyperbolicGCNblock(MessagePassing):
    def __init__(self, in_dim, out_dim, c_in, c_out, dropout_rate=0.5,  add_self_loops=True, fill_value="mean", **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super(hyperbolicGCNblock, self).__init__(node_dim=0, **kwargs)  # "Add" aggregation.
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c_in = c_in
        self.c_out = c_out
        self.dropout_rate = dropout_rate
        self.bias = nn.Parameter(torch.Tensor(out_dim)).cuda()
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim)).cuda()
        self.lin = Linear(in_dim, out_dim, bias=False, weight_initializer="glorot").cuda()
        self.reset_parameters()
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)    
        self.lin.reset_parameters()
        
       
    def forward(self, x, edge_index, edge_attr=None,size=None):
        drop_weight = F.dropout(self.weight, self.dropout_rate, training=self.training).cuda()
        curvature_in = F.softplus(self.c_in)
        curvature_out = F.softplus(self.c_out)
        
        mv = mobius_matvec(drop_weight, x, curvature_in )
        res = proj(mv, curvature_in)
        
        bias = proj_tan0(self.bias.view(1,-1), curvature_in)
        hyp_bias = expmap0(bias, curvature_in)
        hyp_bias = proj(hyp_bias, curvature_in)
        res = mobius_add(res, hyp_bias, c = curvature_in)
        res = proj(res, curvature_in)
        x_tangent = logmap0(res, c= curvature_in)
        x_src = x_dst = self.lin(x_tangent).view(-1,1,self.out_dim)
        x= (x_src, x_dst)
        
        alpha_src = (x_src).sum(dim=-1)
        
        alpha_dst = None if x_dst is None else (x_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")
        alpha = self.edge_updater(edge_index, alpha = alpha, edge_attr = edge_attr)
        
        # out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        out = self.propagate(edge_index, x=x,alpha=alpha,size=size)
        out = out.view(-1, self.out_dim)
        out = proj(expmap(res, out,c=curvature_in),c=curvature_in )
        
        out = F.relu(logmap0(out, c=curvature_in))
        out = proj_tan0(out, c=self.c_out)
        out = proj(expmap0(out, c=curvature_out), curvature_out)
        
        
        return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        
        
        
        return alpha.unsqueeze(-1) * x_j

    # def update(self, aggr_out):
    #     # aggr_out has shape [N, out_channels]

    #     # Step 5: Return new node embeddings.
    #     return aggr_out
    
    
    
    
    
eps = {torch.float32: 1e-7, torch.float64: 1e-15}
min_norm = 1e-5
max_norm = 1e6

def mobius_add(x, y, c):
    u = logmap0(y, c)
    v = ptransp0(x, u, c)
    return expmap(v, x, c)

def mobius_matvec(m, x, c):
    
    u = logmap0(x, c)
    
    
    
    
    mu = u @ m
    
    
    
    return expmap0(mu, c)


def ptransp0(x, u, c):
    K = 1. / c
    sqrtK = K ** 0.5
    x0 = x.narrow(-1, 0, 1)
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=min_norm)
    y_normalized = y / y_norm
    v = torch.ones_like(x)
    v[:, 0:1] = - y_norm 
    v[:, 1:] = (sqrtK - x0) * y_normalized
    alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
    res = u - alpha * v
    return proj_tan(res, x, c)

def expmap(u, x, c):
    K = 1. / c
    sqrtK = K ** 0.5
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, max=max_norm)
    theta = normu / sqrtK
    theta = torch.clamp(theta, min=min_norm)
    result = cosh(theta) * x + sinh(theta) * u / theta
    return proj(result, c)

def minkowski_dot(x, y, keepdim=True):
    res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
    if keepdim:
        res = res.view(res.shape + (1,))
    return res
 
def minkowski_norm(u, keepdim=True):
    dot = minkowski_dot(u, u, keepdim=keepdim)
    return torch.sqrt(torch.clamp(dot, min=eps[u.dtype]))    
def proj( x, c):
    K = 1. / c
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    vals = torch.zeros_like(x)
    
    vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=eps[x.dtype]))
    return vals + mask * x

def proj_tan0( u, c):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[:, 0:1] = narrowed
    return u - vals


def expmap0( u, c):
    K = 1. / c
    sqrtK = K ** 0.5
    d = u.size(-1) - 1
    x = u.narrow(-1, 1, d).view(-1, d)
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    
    x_norm = torch.clamp(x_norm, min=min_norm)
    
    theta = x_norm / sqrtK
    
    
    res = torch.ones_like(u)
    res[:, 0:1] = sqrtK * cosh(theta)
    res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
    return proj(res, c)



def logmap0( x, c):
    K = 1. / c
    sqrtK = K ** 0.5
    d = x.size(-1) - 1

    y = x.narrow(-1, 1, d).view(-1, d)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    y_norm = torch.clamp(y_norm, min=min_norm)
    res = torch.zeros_like(x)
    theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + eps[x.dtype])
    res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
    return res


def proj_tan(u, x, c):
    K = 1. / c
    d = x.size(1) - 1
    ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
    mask = torch.ones_like(u)
    mask[:, 0] = 0
    vals = torch.zeros_like(u)
    vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=eps[x.dtype])
    return vals + mask * u