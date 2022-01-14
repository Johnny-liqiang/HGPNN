import math
import torch.nn as nn
import torch.nn.functional as F
from data import *
from torch.nn.init import ones_
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import softmax


class HGT(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, in_sag, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // self.n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.initializer = glorot
        self.in_sag = in_sag

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mappings = nn.ModuleList()

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            self.mappings.append(nn.Linear(in_dim, out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, self.n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, self.n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x, edge_index, node_type, edge_type):
        return self.propagate(edge_index, node_feature=x, node_type=node_type, edge_type=edge_type)

    def message(self, edge_index_i, node_feature_i, node_feature_j, node_type_i, node_type_j, edge_type):
        data_size = edge_index_i.size(0)

        res_att = torch.zeros(data_size, self.n_heads).to(node_feature_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_feature_i.device)

        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type))
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]

            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]

                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    idx = idx.squeeze()
                    if idx.sum() == 0:
                        continue

                    target_node_vec = node_feature_i[idx]
                    source_node_vec = node_feature_j[idx]

                    '''
                        Step 1: Heterogeneous Mutual Attention
                    '''
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k).to(node_feature_i.device)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k).to(node_feature_i.device)
                    '''
                        batch matrix-matrix product: (n, a, b) * (n, b, c) = (n, a, c)

                        (n_edge, n_heads, d_k) -> (n_heads, n_edge, d_k)

                        (n_heads, n_edge, d_k) * (n_heads, d_k, d_k)
                        -> (n_heads, n_edge, d_k)
                        -> (n_edge, n_heads, d_k)
                    '''
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    '''
                        Step 2: Heterogeneous Message Passing
                    '''
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        '''
            (data_size, n_heads) -> (data_size, n_heads, 1)

            (data_size, n_heads, d_k) * (data_size, n_heads, 1) = (data_size, n_heads, d_k)
        '''
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_feature, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_feature.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type)).squeeze()
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])
            '''
                Add skip connection with learnable weight self.skip[t_id]
            '''
            if self.in_sag:
                res[idx] = trans_out * alpha + self.mappings[target_type](node_feature[idx]) * (1 - alpha)
            else:
                res[idx] = trans_out * alpha + node_feature[idx] * (1 - alpha)

        return res

    def reset_parameters(self):
        ones_(self.relation_pri)
        ones_(self.skip)
        self.initializer(self.relation_att)
        self.initializer(self.relation_msg)
        for k, q, v, a, norm, mapping in zip(self.k_linears, self.q_linears, self.v_linears, self.a_linears, self.norms,
                                             self.mappings):
            k.reset_parameters()
            q.reset_parameters()
            v.reset_parameters()
            a.reset_parameters()
            norm.reset_parameters()
            mapping.reset_parameters()


class MySAGPooling(nn.Module):
    def __init__(self, in_channels, out_dim, num_types, num_relations, dropout=0.2, ratio=0.5, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh):
        super(MySAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = HGT(in_channels, out_dim, num_types, num_relations, n_heads=1, in_sag=True, dropout=dropout)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, node_type, edge_type, batch):
        num_nodes = x.shape[0]

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        res = self.gnn(x, edge_index, node_type, edge_type)

        score = res.view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        batch = batch[perm]

        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[perm] = 1
        edge_mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        out_edge_type = edge_type[edge_mask]

        edge_index, _ = filter_adj(edge_index, None, perm, num_nodes=score.size(0))

        return x, edge_index, node_type[perm], out_edge_type, batch
