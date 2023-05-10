
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer. 
    """

    def __init__(self, in_features, out_features, edge_dim=3, aggregate='sum', dropout=0., use_relu=True, bias=False):
        '''
        :param in/out_features: scalar of channels for node embedding
        :param edge_dim: dim of edge type, virtual type not included
        '''
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(
            self.edge_dim, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        typically d=9 e=3
        :return:
        updated x with shape (batch, N, d)
        '''
        x = F.dropout(x, p=self.dropout, training=self.training)  # (b, N, d)

        batch_size = x.size(0)

        # transform
        support = torch.einsum('bid, edh-> beih', x, self.weight)
        output = torch.einsum('beij, bejh-> beih', adj,
                              support)  # (batch, e, N, d)

        if self.bias is not None:
            output += self.bias
        if self.act is not None:
            output = self.act(output)  # (b, E, N, d)
        output = output.view(batch_size, self.edge_dim, x.size(
            1), self.out_features)  # (b, E, N, d)

        if self.aggregate == 'sum':
            # sum pooling #(b, N, d)
            node_embedding = torch.sum(output, dim=1, keepdim=False)
        elif self.aggregate == 'max':
            # max pooling  #(b, N, d)
            node_embedding = torch.max(output, dim=1, keepdim=False)
        elif self.aggregate == 'mean':
            # mean pooling #(b, N, d)
            node_embedding = torch.mean(output, dim=1, keepdim=False)
        elif self.aggregate == 'concat':
            #! implementation wrong
            node_embedding = torch.cat(torch.split(
                output, dim=1, split_size_or_sections=1), dim=3)  # (b, 1, n, d*e)
            node_embedding = torch.squeeze(
                node_embedding, dim=1)  # (b, n, d*e)
        else:
            print('GCN aggregate error!')
        return node_embedding

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class RGCN(nn.Module):
    def __init__(self, nfeat, nhid=128, nout=128, edge_dim=3, num_layers=3, dropout=0., normalization=False):
        '''
        :num_layars: the number of layers in each R-GCN
        '''
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.normalization = normalization

        self.emb = Linear(nfeat, nfeat, bias=False) 
        #self.bn_emb = nn.BatchNorm2d(8)

        self.gc1 = RelationGraphConvolution(
            nfeat, nhid, edge_dim=self.edge_dim, aggregate='sum', use_relu=True, dropout=self.dropout, bias=False)
        # if self.normalization:
        #    self.bn1 = nn.BatchNorm2d(nhid)

        self.gc2 = nn.ModuleList([RelationGraphConvolution(nhid, nhid, edge_dim=self.edge_dim, aggregate='sum',
                                                           use_relu=True, dropout=self.dropout, bias=False)
                                  for i in range(self.num_layers-2)])
        # if self.normalization:
        #    self.bn2 = nn.ModuleList([nn.BatchNorm2d(nhid) for i in range(self.num_layers-2)])

        self.gc3 = RelationGraphConvolution(
            nhid, nout, edge_dim=self.edge_dim, aggregate='sum', use_relu=False, dropout=self.dropout, bias=False)
        # if self.normalization
        #    self.bn3 = nn.BatchNorm2d(nout)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        :return:
        '''
        # TODO: Add normalization for adacency matrix
        # embedding layer
        x = self.emb(x)
        # if self.normalization:
        #    x = self.bn_emb(x.transpose(0, 3, 1, 2))
        #    x = x.transpose(0, 2, 3, 1)

        # first GCN layer
        x = self.gc1(x, adj)
        # if self.normalization:
        #    x = self.bn1(x.transpose(0, 3, 1, 2))
        #    x = x.transpose(0, 2, 3, 1)

        # hidden GCN layer(s)
        for i in range(self.num_layers-2):
            x = self.gc2[i](x, adj)  # (#node, #class)
            # if self.normalization:
            #    x = self.bn2[i](x.transpose(0, 3, 1, 2))
            #    x = x.transpose(0, 2, 3, 1)

        # last GCN layer
        x = self.gc3(x, adj)  # (batch, N, d)
        # check here: bn for last layer seem to be necessary
        #x = self.bn3(x.transpose(0, 3, 1, 2))
        #x = x.transpose(0, 2, 3, 1)

        # return node embedding
        return x

# TODO: Try sample dependent initialization.!!
# TODO: Try different st function (sigmoid, softplus, exp, spine, flow++)
class ST_Net_Sigmoid(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Sigmoid, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)


    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.sigmoid(s + self.sigmoid_shift)
        s = self.rescale2(s) # linear scale seems important, similar to learnable prior..
        return s, t


class ST_Net_Exp(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            #self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            #self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        #x = self.rescale1(x)
        #x = x.view((-1, self.output_dim*2))
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        #s = self.sigmoid(s + self.sigmoid_shift)
        #s = self.rescale2(s) # linear scale seems important, similar to learnable prior..
        return s, t


class ST_Net_Softplus(nn.Module):
    def __init__(self, args, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Softplus, self).__init__()
        self.num_layers = num_layers  # unused
        self.args = args
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.linear3 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale_channel(self.output_dim))

        else:
            self.rescale1 = Rescale_channel(self.output_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear3.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)
            nn.init.constant_(self.linear3.bias, 0.)


    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = F.tanh(self.linear2(F.relu(self.linear1(x))))
        x = self.linear3(x)
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.softplus(s)
        s = self.rescale1(s) # linear scale seems important, similar to learnable prior..
        return s, t


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x


class Rescale_channel(nn.Module):
    def __init__(self, num_channels):
        super(Rescale_channel, self).__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.zeros([num_channels]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            raise RuntimeError('Rescale factor has NaN entries')
            
        x = torch.exp(self.weight) * x
        return x


class MaskedGraphAF(nn.Module):
    def __init__(self, mask_node, mask_edge, index_select_edge, num_flow_layer, graph_size,
                 num_node_type, num_edge_type, args, nhid=128, nout=128):
        '''
        :param index_nod_edg:
        :param num_edge_type, virtual type included
        '''
        super(MaskedGraphAF, self).__init__()
        self.repeat_num = mask_node.size(0)
        self.graph_size = graph_size
        self.num_node_type = num_node_type
        self.num_edge_type = num_edge_type
        self.args = args
        self.is_batchNorm = self.args.is_bn

        self.mask_node = nn.Parameter(mask_node.view(1, self.repeat_num, graph_size, 1), requires_grad=False)  # (1, repeat_num, n, 1)
        self.mask_edge = nn.Parameter(mask_edge.view(1, self.repeat_num, 1, graph_size, graph_size), requires_grad=False)  # (1, repeat_num, 1, n, n)

        self.index_select_edge = nn.Parameter(index_select_edge, requires_grad=False)  # (edge_step_length, 2)

        self.emb_size = nout
        self.hid_size = nhid
        self.num_flow_layer = num_flow_layer

        self.rgcn = RGCN(num_node_type, nhid=self.hid_size, nout=self.emb_size, edge_dim=self.num_edge_type-1,
                         num_layers=self.args.gcn_layer, dropout=0., normalization=False)

        if self.is_batchNorm:
            self.batchNorm = nn.BatchNorm1d(nout)

        self.st_net_fn_dict = {'sigmoid': ST_Net_Sigmoid,
                               'exp': ST_Net_Exp,
                               'softplus': ST_Net_Softplus,
                              }
        assert self.args.st_type in ['sigmoid', 'exp', 'softplus'], 'unsupported st_type, choices are [sigmoid, exp, softplus, ]'
        st_net_fn = self.st_net_fn_dict[self.args.st_type]

        self.node_st_net = nn.ModuleList([st_net_fn(self.args, nout, self.num_node_type, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])

        self.edge_st_net = nn.ModuleList([st_net_fn(self.args, nout*3, self.num_edge_type, hid_dim=nhid, bias=True,
                                                 scale_weight_norm=self.args.scale_weight_norm, sigmoid_shift=self.args.sigmoid_shift, 
                                                 apply_batch_norm=self.args.is_bn_before) for i in range(num_flow_layer)])


    def forward(self, x, adj, x_deq, adj_deq):
        '''
        :param x:   (batch, N, 14)
        :param adj: (batch, 4, N, N)

        :param x_deq: (batch, N, 14)
        :param adj_deq:  (batch, edge_num, 4)
        :return:
        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        graph_emb_node, graph_node_emb_edge = self._get_embs(x, adj)

        x_deq = x_deq.view(-1, self.num_node_type)  # (batch *N, 14)
        adj_deq = adj_deq.view(-1, self.num_edge_type) # (batch*(repeat_num-N), 4)


        for i in range(self.num_flow_layer): #12
            # update x_deq
            node_s, node_t = self.node_st_net[i](graph_emb_node)
            if self.args.st_type == 'sigmoid':
                x_deq = x_deq * node_s + node_t
            elif self.args.st_type == 'exp':
                node_s = node_s.exp() #TODO: rescale is exponential, check it.
                x_deq = (x_deq + node_t) * node_s
            elif self.args.st_type == 'softplus':
                x_deq = (x_deq + node_t) * node_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(x_deq).any():
                raise RuntimeError(
                    'x_deq has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()

            # update adj_deq
            edge_s, edge_t = self.edge_st_net[i](graph_node_emb_edge)
            if self.args.st_type == 'sigmoid':
                adj_deq = adj_deq * edge_s + edge_t
            elif self.args.st_type == 'exp':
                edge_s = edge_s.exp()
                adj_deq = (adj_deq + edge_t) * edge_s
            elif self.args.st_type == 'softplus':
                adj_deq = (adj_deq + edge_t) * edge_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(adj_deq).any():
                raise RuntimeError(
                    'adj_deq has NaN entries after transformation at layer %d' % i)
            if i == 0:
                adj_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                adj_log_jacob += (torch.abs(edge_s) + 1e-20).log()

        x_deq = x_deq.view(batch_size, -1)  # (batch, N * 9)
        adj_deq = adj_deq.view(batch_size, -1)  # (batch, (repeat_num-N) * 4)

        x_log_jacob = x_log_jacob.view(batch_size, -1).sum(-1)  # (batch)
        adj_log_jacob = adj_log_jacob.view(batch_size, -1).sum(-1)  # (batch)
        return [x_deq, adj_deq], [x_log_jacob, adj_log_jacob]


    def forward_rl_node(self, x, adj, x_cont):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 9)
        Returns:
            x_cont: shape (batch, 9)
            x_log_jacob: shape (batch, )
        """
        embs = self._get_embs_node(x, adj) # (batch, d)
        for i in range(self.num_flow_layer):
            node_s, node_t = self.node_st_net[i](embs)

            if self.args.st_type == 'sigmoid':
                x_cont = x_cont * node_s + node_t
            elif self.args.st_type == 'exp':
                node_s = node_s.exp()
                x_cont = (x_cont + node_t) * node_s
            elif self.args.st_type == 'softplus':
                x_cont = (x_cont + node_t) * node_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(
                    'x_cont has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(node_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(node_s) + 1e-20).log()            

        x_log_jacob = x_log_jacob.sum(-1)  # (batch)

        return x_cont, x_log_jacob



    def forward_rl_edge(self, x, adj, x_cont, index):
        """
        Args:
            x: shape (batch, N, 9)
            adj: shape (batch, 4, N, N)
            x_cont: shape (batch, 4)
            index: shape (batch, 2)
        Returns:
            x_cont: shape (batch, 4)
            x_log_jacob: shape (batch, )            
        """
        embs = self._get_embs_edge(x, adj, index) # (batch, 3d)

        for i in range(self.num_flow_layer):
            edge_s, edge_t = self.edge_st_net[i](embs)

            if self.args.st_type == 'sigmoid':
                x_cont = x_cont * edge_s + edge_t
            elif self.args.st_type == 'exp':
                edge_s = edge_s.exp()
                x_cont = (x_cont + edge_t) * edge_s
            elif self.args.st_type == 'softplus':
                x_cont = (x_cont + edge_t) * edge_s
            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)

            if torch.isnan(x_cont).any():
                raise RuntimeError(
                    'x_cont has NaN entries after transformation at layer %d' % i)

            if i == 0:
                x_log_jacob = (torch.abs(edge_s) + 1e-20).log()
            else:
                x_log_jacob += (torch.abs(edge_s) + 1e-20).log()            

        x_log_jacob = x_log_jacob.sum(-1)  # (batch)
        
        return x_cont, x_log_jacob        


    def reverse(self, x, adj, latent, mode, edge_index=None):
        '''
        Args:
            x: generated subgraph node features so far with shape (1, N, 14), some part of the x is masked
            adj: generated subgraph adacency features so far with shape (1, 4, N, N) some part of the adj is masked
            latent: sample latent vector with shape (1, 14) (mode == 0) or (1, 4) (mode == 1)
            mode: generation mode. if mode == 0, generate a new node, if mode == 1, generate a new edge
            edge_index [1, 2]

        Returns:
            out: generated node/edge features with shape (1, 14) (mode == 0) or (1, 4) , (mode == 1)
        '''

        assert mode == 0 or edge_index is not None, 'if you want to generate edge, you must specify edge_index'
        assert x.size(0) == 1
        assert adj.size(0) == 1
        assert edge_index is None or (edge_index.size(0) == 1 and edge_index.size(1) == 2)
        
        if mode == 0: #(1, 9)
            st_net = self.node_st_net
            #emb = graph_emb
            emb = self._get_embs_node(x, adj)

        else:  # mode == 1
            st_net = self.edge_st_net
            emb = self._get_embs_edge(x, adj, edge_index)            

        for i in reversed(range(self.num_flow_layer)):
            s, t = st_net[i](emb)
            if self.args.st_type == 'sigmoid':
                latent = (latent - t) / s
            elif self.args.st_type == 'exp':
                s = s.exp()
                latent = (latent / s) - t
            elif self.args.st_type == 'softplus':
                latent = (latent / s) - t

            else:
                raise ValueError('unsupported st type: (%s)' % self.args.st_type)     

        return latent

    def _get_embs_node(self, x, adj):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 14)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
        Returns:
            graph embedding for updating node features with shape (batch, d)
        """

        batch_size = x.size(0)
        adj = adj[:, :3] # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj) # (batch, N, d)
        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)
        
        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous() # (batch, d)
        return graph_emb

    def _get_embs_edge(self, x, adj, index):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 14)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
            index: link prediction index with shape (batch, 2)
        Returns:
            Embedding(concatenate graph embedding, edge start node embedding and edge end node embedding) 
                for updating edge features with shape (batch, 3d)
        """

        batch_size = x.size(0)
        assert batch_size == index.size(0)

        adj = adj[:, :3] # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj) # (batch, N, d)
        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2) # (batch, N, d)

        graph_emb = torch.sum(node_emb, dim = 1, keepdim=False).contiguous().view(batch_size, 1, -1) # (batch, 1, d)

        index = index.view(batch_size, -1, 1).repeat(1, 1, self.emb_size) # (batch, 2, d)
        graph_node_emb = torch.cat((torch.gather(node_emb, dim=1, index=index), 
                                        graph_emb),dim=1)  # (batch_size, 3, d)
        graph_node_emb = graph_node_emb.view(batch_size, -1) # (batch_size, 3d)
        return graph_node_emb


    def _get_embs(self, x, adj):
        '''
        :param x of shape (batch, N, 14)
        :param adj of shape (batch, 4, N, N)
        :return: inputs for st_net_node and st_net_edge
        graph_emb_node of shape (batch*N, d)
        graph_emb_edge of shape (batch*(repeat-N), 3d)

        '''
        # inputs for RelGCNs
        batch_size = x.size(0)
        adj = adj[:, :3] # (batch, 3, N, N) TODO: check whether we have to use the 4-th slices(virtual bond) or not

        x = torch.where(self.mask_node, x.unsqueeze(1).repeat(1, self.repeat_num, 1, 1), torch.zeros([1]).cuda()).view(
            -1, self.graph_size, self.num_node_type)  # (batch*repeat_num, N, 14)

        adj = torch.where(self.mask_edge, adj.unsqueeze(1).repeat(1, self.repeat_num, 1, 1, 1),
                          torch.zeros([1]).cuda()).view(
            -1, self.num_edge_type - 1, self.graph_size, self.graph_size)  # (batch*repeat_num, 3, N, N)

        node_emb = self.rgcn(x, adj)  # (batch*repeat_num, N, d)

        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch*repeat_num, N, d)

        node_emb = node_emb.view(batch_size, self.repeat_num, self.graph_size, -1) # (batch, repeat_num, N, d)


        graph_emb = torch.sum(node_emb, dim=2, keepdim=False) # (batch, repeat_num, d)

        #  input for st_net_node
        graph_emb_node = graph_emb[:, :self.graph_size].contiguous() # (batch, N, d)
        graph_emb_node = graph_emb_node.view(batch_size * self.graph_size, -1)  # (batch*N, d)

        # input for st_net_edge
        graph_emb_edge = graph_emb[:, self.graph_size:].contiguous() # (batch, repeat_num-N, d)
        graph_emb_edge = graph_emb_edge.unsqueeze(2)  # (batch, repeat_num-N, 1, d)

        all_node_emb_edge = node_emb[:, self.graph_size:] # (batch, repeat_num-N, N, d)

        index = self.index_select_edge.view(1, -1, 2, 1).repeat(batch_size, 1, 1,
                                        self.emb_size)  # (batch_size, repeat_num-N, 2, d)


        graph_node_emb_edge = torch.cat((torch.gather(all_node_emb_edge, dim=2, index=index), 
                                        graph_emb_edge),dim=2)  # (batch_size, repeat_num-N, 3, d)

        graph_node_emb_edge = graph_node_emb_edge.view(batch_size * (self.repeat_num - self.graph_size),
                                        -1)  # (batch_size * (repeat_num-N), 3*d)

        return graph_emb_node, graph_node_emb_edge


if __name__ == '__main__':
    data = torch.load('debug_data.pt')
    maskgaf = MaskedGraphAF(mask_node=data['node_masks'], mask_edge=data['adj_masks'],
                            index_nod_edg=data['is_node_update_mask'],
                            index_select_edge=data['link_prediction'].long(), num_flow_layer=3, graph_size=45,
                            num_node_type=9, num_edge_type=4, nhid=128, nout=128)
    deq, jacob = maskgaf(data['node_feature'], data['adj_feature'], data['node_feature_cont'],
                         data['adj_feature_cont'].permute(0, 2, 1).contiguous())

    print(data['node_feature'][0][2])
    print(deq[0].size())
    print(deq[1].size())

    print(jacob[0].size())
    print(jacob[1].size())
    # print(deq[0][0])
    # print(deq[0][1])
    # print(deq[0][2])
    # print(deq[0][3])
    # print(deq[0][4])

    # print(deq[0][45])
    # print(deq[0][46])
    # print(deq[0][47])
