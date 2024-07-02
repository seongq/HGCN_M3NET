import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from mathutils import cosh, sinh, arcosh
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
import math
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add


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
class hyperbolicGCN_hypergraph(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels,dropout_rate = 0.5,curvature=0.1):
        super(hyperbolicGCN_hypergraph, self).__init__()
        self.num_layers = num_layers
        self.base_curvature = 0.1
        self.curvatures = nn.ParameterList([nn.Parameter(torch.Tensor([curvature])).cuda() for _ in range(num_layers+1)])
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
       
        hgc_layers = []
        for i in range(self.num_layers):
            c_in, c_out = self.curvatures[i], self.curvatures[i+1]
            in_dim = in_channels
            out_dim = out_channels
            hgc_layers.append(hyperbolicGCNblock_hypergraph(in_dim, out_dim, c_in, c_out, self.dropout_rate))
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
        
    def forward(self,x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, EW_weight=None, dia_len = None):
        encoded = self.encoder(x,hyperedge_index)
        
        for layer in self.hgc_layers:
            
            encoded = layer(encoded, hyperedge_index)
            
        last_hyp_rep = encoded
        decoded = self.decoder(last_hyp_rep, hyperedge_index)
        return decoded
        
class hyperbolicGCNblock_hypergraph(MessagePassing):
    def __init__(self, in_dim, out_dim, c_in, c_out, dropout_rate=0.5,  use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,  **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super(hyperbolicGCNblock_hypergraph, self).__init__(node_dim=0, **kwargs)  # "Add" aggregation.
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c_in = c_in
        self.c_out = c_out
        self.dropout_rate = dropout_rate


        self.use_attention = use_attention
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_dim, heads * out_dim))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_dim))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.edgeweight = Parameter(torch.Tensor(in_dim, out_dim))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.edgefc = torch.nn.Linear(in_dim, out_dim)
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
       
    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, EW_weight=None, dia_len = None):
        curvature_in = F.softplus(self.c_in)
        curvature_out = F.softplus(self.c_out)
        num_nodes, num_edges = x.size(0), 0 
        x_tangent = proj_tan0(logmap0(x, c= curvature_in), c=curvature_in)
        x = x_tangent
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)
       
       
       
        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_dim)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_dim)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)      #[num_nodes]
        D = 1.0 / D                                                         # all 0.5 if hyperedge_weight is None
        D[D == float("inf")] = 0
        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                        hyperedge_index[1], dim=0, dim_size=num_edges)      #[num_edges]
        B = 1.0 / B
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,#hyperedge_attr[hyperedge_index[1]],  
                             size=(num_nodes, num_edges))                   #num_edges,1,100
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_nodes,num_edges))                   #num_nodes,1,100
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            bias = proj_tan0(self.bias.view(1,-1), c=curvature_in)
            hyp_bias = expmap0(bias, c=curvature_in)
            hyp_bias = proj(hyp_bias, c=curvature_in)
            res = proj(out, c=curvature_in)
            res = mobius_add(res, hyp_bias, c=curvature_in)
            res = proj(res, c=curvature_in)
            out = res
            out = torch.nn.LeakyReLU()(logmap0(out,c=curvature_in))
            out = proj_tan0(out,c=curvature_in)
            
        
        
        out = expmap0(out, curvature_out)
        out = proj(out,curvature_out)
        
        return out

   
    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_dim

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j      
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
  
    
    
    
    
eps = {torch.float32: 1e-7, torch.float64: 1e-15}
min_norm = 1e-5
max_norm = 1e6

def mobius_add(x, y, c):
    u = logmap0(y, c)
    v = ptransp0(x, u, c)
    return expmap(v, x, c)

def mobius_matvec(m, x, c):
    
    u = logmap0(x, c)
    # print("u size",u, )
    # print("m", m)
    # print("u size", u.size())
    # print("m size", m.size())
    mu = u @ m
    # print("mu size", mu.size())
    # print("mu는 이것다", mu)
    
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
    # print(x_norm)
    x_norm = torch.clamp(x_norm, min=min_norm)
    
    theta = x_norm / sqrtK
    # print(theta)
    # print(sqrtK)
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