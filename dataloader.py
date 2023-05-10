import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset


def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next = next + neighbor
        output = output + next
        start = next
    return output

class CondDataset(Dataset):
    def __init__(self, full_nodes, full_edges, frag_nodes,frag_edges,
                            linker_len, exit_point, v_to_keep, full_smi):        
        self.n_molecule = full_nodes.shape[0]
        self.full_nodes = full_nodes
        self.full_edges = full_edges
        self.frag_nodes = frag_nodes
        self.frag_edges = frag_edges
        self.linker_sizes = linker_len
        self.exit_point = exit_point
        self.v_to_keep = v_to_keep
        self.full_smi = full_smi

        self.atom_list ={0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'}
        self.max_size = self.full_nodes.shape[1] # N
        self.node_dim = len(self.atom_list) #14

    def __len__(self):
        return self.n_molecule

    def __getitem__(self, idx):
        full_nodes = self.full_nodes[idx] #(N,14)
        full_edges = self.full_edges[idx] #(4, N, N)
        frag_nodes = self.frag_nodes[idx] #(N,14)
        frag_edges = self.frag_edges[idx] #(4, N, N)
        linker_len = self.linker_sizes[idx]
        exit_point = self.exit_point[idx] #[a,b]
        v_to_keep = self.v_to_keep[idx]
        full_smi = self.full_smi[idx]
        return {'full_nodes': torch.Tensor(full_nodes), 
                'full_edges': torch.Tensor(full_edges),
                'frag_nodes': torch.Tensor(frag_nodes), 
                'frag_edges': torch.Tensor(frag_edges),
                'linker_len':linker_len,
                'exit_point':exit_point,
                'v_to_keep':v_to_keep,
                'full_smi':full_smi
                }


class DataIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        
    def __next__(self):
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data