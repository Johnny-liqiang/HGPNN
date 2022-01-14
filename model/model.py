from layers import *
from util import count_parameters
import numpy as np


class MyModel(nn.Module):
    def __init__(self, in_dim, emb_dim, out_dim, num_types, num_relations, n_node_in_graph=11, dropout=0.2):
        super().__init__()
        if emb_dim is not None:
            self.emb = nn.Linear(in_dim, emb_dim)
        else:
            self.emb = None
            emb_dim = in_dim

        self.lin_dim = lin_dim
        self.lin2_dim = lin2_dim
        self.lin_dropout = lin_dropout
        if lin_dim is not None:
            if lin2_dim is None:
                self.lin2 = nn.Linear(lin_dim, out_dim)
            else:
                self.lin2 = nn.Linear(lin_dim, lin2_dim)
                self.lin3 = nn.Linear(lin2_dim, out_dim)

        self.readout_mode = readout_mode
        self.HGTs = nn.ModuleList()
        self.poolings = nn.ModuleList()

        for _ in range(n_HGTs):
            self.HGTs.append(HGT(emb_dim, emb_dim, num_types, num_relations, n_heads=n_heads, in_sag=False,
                                 dropout=dropout))

        for _ in range(n_SAGs):
            self.poolings.append(
                MySAGPooling(emb_dim, 1, num_types, num_relations, dropout=dropout, ratio=pooling_k))

        if readout_mode == 'cat':
            readout_dim = np.ceil((pooling_k ** n_SAGs) * n_node_in_graph).astype('int') * emb_dim
        elif readout_mode == 'sum':
            readout_dim = emb_dim * 2
        else:
            raise RuntimeError('Unexpected readout mode!')

        self.lin = nn.Linear(readout_dim, out_dim) if lin_dim is None else nn.Linear(readout_dim, lin_dim)

    def forward(self, graph):
        if self.emb is not None:
            res = self.emb(graph.x)
        else:
            res = graph.x

        for hgt in self.HGTs:
            res = hgt(res, graph.edge_index, graph.node_type, graph.edge_type)

        edge_index = graph.edge_index
        node_type = graph.node_type
        edge_type = graph.edge_type
        batch = graph.batch
        for pooling in self.poolings:
            res, edge_index, node_type, edge_type, batch = pooling(res, edge_index, node_type, edge_type, batch)

        if readout_mode == 'cat':
            read_out = res.view(graph.num_graphs, -1)
        elif readout_mode == 'sum':
            features = res.view(graph.num_graphs, -1, res.shape[-1])
            read_out = torch.cat([features.max(1)[0], features.mean(1)], dim=-1)
        else:
            raise RuntimeError('Unexpected readout mode!')

        out = self.lin(read_out)
        if self.lin_dim is not None:
            out = F.dropout(F.leaky_relu(out), p=self.lin_dropout)
            out = self.lin2(out)
            if self.lin2_dim is not None:
                out = F.dropout(F.leaky_relu(out), p=self.lin_dropout)
                out = self.lin3(out)

        return F.log_softmax(out, dim=-1)

    def reset_parameters(self):
        if self.emb is not None:
            self.emb.reset_parameters()
        if self.lin_dim is not None:
            self.lin2.reset_parameters()
            if self.lin2_dim is not None:
                self.lin3.reset_parameters()
        self.lin.reset_parameters()
        for hgt in self.HGTs:
            hgt.reset_parameters()
        for pooling in self.poolings:
            pooling.reset_parameters()

    def show_parameter_num(self):
        cnt = 0
        if self.emb is not None:
            cnt += count_parameters(self.emb, 'emb')
        for i, h in enumerate(self.HGTs):
            cnt += count_parameters(h, f'hgt{i}')
        for i, p in enumerate(self.poolings):
            cnt += count_parameters(p, f'pooling{i}')
        if self.lin2_dim is not None:
            cnt += count_parameters(self.lin, 'lin1')
            cnt += count_parameters(self.lin2, 'lin2')
            cnt += count_parameters(self.lin3, 'classifier')
        elif self.lin_dim is not None:
            cnt += count_parameters(self.lin, 'lin')
            cnt += count_parameters(self.lin2, 'classifier')
        else:
            cnt += count_parameters(self.lin, 'classifier')
        print(f'{cnt:,} total parameters.')
        print('----------------------------------------')
