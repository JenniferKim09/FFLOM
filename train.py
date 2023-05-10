#coding: utf-8
from time import time
import argparse
import numpy as np
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model import GraphFlowModel
from dataloader import CondDataset
import environment as env


def save_model(model, optimizer, args, var_list, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )    


def restore_model(model, args, epoch=None):
    if epoch is None:
        restore_path = os.path.join(args.save_path, 'checkpoint')
        print('restore from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('restore from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def read_molecules(path):
    print('reading data from %s' % path)
    full_nodes = np.load(path + 'full_nodes.npy')
    full_edges = np.load(path + 'full_edges.npy')
    frag_nodes = np.load(path + 'frag_nodes.npy')
    frag_edges = np.load(path + 'frag_edges.npy')
    v_to_keep = np.load(path + 'v_to_keep.npy')
    exit_point = np.load(path + 'exit_point.npy')
    full_smi = np.load(path + 'full_smi.npy')
    length = np.load(path + 'gen_len.npy')

    f = open(path + 'config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    return full_nodes, full_edges, frag_nodes,frag_edges, length, v_to_keep, exit_point, full_smi, data_config

    


class Trainer(object):
    def __init__(self, dataloader, data_config, args, all_train_smiles=None):
        self.dataloader = dataloader
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] #- 1 # exclude padding dim.
        self.bond_dim = self.data_config['bond_dim']
       
        
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_loss = 100.0
        self.start_epoch = 0
        if self.args.cuda:
            self._model = self._model.cuda()
    

    def initialize_from_checkpoint(self):
        checkpoint = torch.load(self.args.init_checkpoint)
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)


    def fit(self):        
        t_total = time()
        total_loss = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch

        print('start fitting.')
        for epoch in range(self.args.epochs):
            epoch_loss = self.train_epoch(epoch + start_epoch)
            total_loss.append(epoch_loss)
            if args.compare_loss==True:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    var_list = {'cur_epoch': epoch + start_epoch,
                                'best_loss': best_loss,
                            }
                    save_model(self._model, self._optimizer, self.args, var_list, epoch=epoch + start_epoch)
            else:
                var_list = {'cur_epoch': epoch + start_epoch,
                            'best_loss': best_loss,
                            }
                save_model(self._model, self._optimizer, self.args, var_list, epoch=epoch + start_epoch)


        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))



    def train_epoch(self, epoch_cnt):
        t_start = time()
        batch_losses = []
        self._model.train()
        batch_cnt = 0

        for i_batch, batch_data in enumerate(self.dataloader):
            batch_time_s = time()

            self._optimizer.zero_grad()

            batch_cnt += 1
            inp_node_features = batch_data['full_nodes'] #(B, N, node_dim)
            inp_adj_features = batch_data['full_edges'] #(B, 4, N, N)            
            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()
            if self.args.deq_type == 'random': #default
                out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features)

                loss = self._model.log_prob(out_z, out_logdet)


                #TODO: add mask for different molecule size, i.e. do not model the distribution over padding nodes.

            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(inp_node_features, inp_adj_features)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(out_z, out_logdet, out_deq_logp, out_deq_logdet)
                loss = -1. * ((ll_node-ll_deq_node) + (ll_edge-ll_deq_edge))
            else:
                raise ValueError('unsupported dequantization method: (%s)' % self.deq_type)
            if args.warm_up:
                self._optimizer.param_groups[0]['lr'] = 0.002 * min((epoch_cnt+1)**(-0.5), (epoch_cnt+1)*(5**(-1.5)))
            loss.backward()
            self._optimizer.step()


            batch_losses.append(loss.item())

            if batch_cnt % self.args.show_loss_step == 0 or (epoch_cnt == 0 and batch_cnt <= 100):

                print('epoch: %d | step: %d | time: %.5f | loss: %.5f | ln_var: %.5f' % (epoch_cnt, batch_cnt, time() - batch_time_s, batch_losses[-1], ln_var))

        epoch_loss = sum(batch_losses) / len(batch_losses)
        #print(epoch_example)
        print('Epoch: {: d}, loss {:5.5f}, epoch time {:.5f}, lr {:.4f}'.format(epoch_cnt, epoch_loss, time()-t_start, self._optimizer.param_groups[0]['lr']))          
        return epoch_loss

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFLOM model')

    # ******data args******
    parser.add_argument('--path', type=str, help='path of dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
    parser.add_argument('--shuffle', action='store_true', default=True, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=4, help='num works to generate data.')

    # ******model args******
    parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9, help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=12, help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    #TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='exp', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******optimization args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--warm_up', action='store_true', default=True, help='Add warm-up and decay to the learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=True, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=True, help='divide loss by length of latent.')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

    parser.add_argument('--show_loss_step', type=int, default=100)
    parser.add_argument('--compare_loss', action='store_true', default=True, help='Save model only if current loss is lower than ever.')

    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    checkpoint_dir = args.all_save_prefix + 'save_pretrain/%s' % (args.name)
    args.save_path = checkpoint_dir

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_seed(args.seed, args.cuda)

    print(args)

    full_nodes, full_edges, frag_nodes,frag_edges, length, v_to_keep, exit_point, full_smi, data_config = read_molecules(args.path)
    train_dataloader = DataLoader(CondDataset(full_nodes, full_edges, frag_nodes,frag_edges, length, v_to_keep, exit_point, full_smi),
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.num_workers)

    trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=full_smi)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint()

    trainer.fit()

