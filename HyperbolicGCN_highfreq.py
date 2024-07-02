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
class hyperbolicGCN_highfreq(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels,dropout_rate = 0.5,curvature=0.1):
        super(hyperbolicGCN_highfreq, self).__init__()
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
            hgc_layers.append(hyperbolicGCNblock_highfreq(in_dim, out_dim, c_in, c_out, self.dropout_rate))
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
        
class hyperbolicGCNblock_highfreq(MessagePassing):
    def __init__(self, in_dim, out_dim, c_in, c_out, dropout_rate=0.5,  add_self_loops=True,  **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super(hyperbolicGCNblock_highfreq, self).__init__(node_dim=0, **kwargs)  # "Add" aggregation.
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c_in = c_in
        self.c_out = c_out
        self.dropout_rate = dropout_rate


        self.gate = torch.nn.Linear(2*in_dim, 1)
        self.reset_parameters()
        self.add_self_loops = add_self_loops
        
    
        
       
    def forward(self, x, edge_index, edge_attr=None,size=None):
        curvature_in = F.softplus(self.c_in)
        curvature_out = F.softplus(self.c_out)
        x_tangent = logmap0(x, c= curvature_in)
        x_tangent = proj_tan0(x_tangent, c = curvature_in)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out= self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x_tangent)
        
        out = out.view(-1, self.out_dim)
        out = expmap0(out,c=curvature_in)
        out = proj(out, c=curvature_in)

        out = proj_tan0(F.relu(logmap0(out, c=curvature_in)), c=curvature_in)

        out = expmap0(out, c=curvature_out)
        out = proj(out,c=curvature_out)


        return out

   
    def message(self, x_i, x_j, edge_index, size):
        # x_j e.g.[135090, 512]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #h2 = x_i - x_j
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))#e.g.[135090, 1]

        return norm.view(-1, 1) * (x_j) *alpha_g
    
    def update(self, aggr_out):
  
        return aggr_out

  
    
    
    
    
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