import os
import argparse
import json
import numpy as np
import utils
from utils import  to_graph_mol, align_mol_to_frags, align_mol_to_frags_elaboration
import numpy as np
import torch

def preprocess_linker(data):
    processed_data =[]
    error=[]
    for i, (smi_mol, smi_frags, smi_linker) in enumerate([(mol['smi_mol'], mol['smi_frags'], 
                                                                        mol['smi_generate']) for mol in data]):
        (mol_out, mol_in), nodes_to_keep, exit_points = align_mol_to_frags(smi_mol, smi_linker, smi_frags)
        if mol_out == []:
            continue
        nodes_in, edges_in = to_graph_mol(mol_in, 'zinc')
        nodes_out, edges_out = to_graph_mol(mol_out, 'zinc')
        if min(len(edges_in), len(edges_out)) <= 0:
            error.append(i)
            continue
        processed_data.append({
                'graph_in': edges_in,
                'graph_out': edges_out, 
                'node_features_in': nodes_in,
                'node_features_out': nodes_out, 
                'smiles_out': smi_mol,
                'smiles_in': smi_frags,
                'v_to_keep': nodes_to_keep,
                'exit_points': exit_points
            })
    print('error: ' + str(len(error)))
    return processed_data

def preprocess_r(data):
    processed_data =[]
    error=[]
    for i, (smi_mol, smi_frags, smi_elab) in enumerate([(mol['smi_mol'], mol['smi_frags'], 
                                                                        mol['smi_generate']) for mol in data]):
        (mol_out, mol_in), nodes_to_keep, exit_points = align_mol_to_frags_elaboration(smi_mol, smi_frags)
        if mol_out == []:
            continue
        nodes_in, edges_in = to_graph_mol(mol_in, 'zinc')
        nodes_out, edges_out = to_graph_mol(mol_out, 'zinc')
        if min(len(edges_in), len(edges_out)) <= 0:
            error.append(i)
            continue
        processed_data.append({
                'graph_in': edges_in,
                'graph_out': edges_out, 
                'node_features_in': nodes_in,
                'node_features_out': nodes_out, 
                'smiles_out': smi_mol,
                'smiles_in': smi_frags,
                'v_to_keep': nodes_to_keep,
                'exit_points': exit_points
            })
    print('error: ' + str(len(error)))
    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFLOM model')
    parser.add_argument('--data', type=str, help='path of dataset', required=True)
    parser.add_argument('--save_fold', type=str, default='./data_preprocessed/', help='path to save data')
    parser.add_argument('--name', type=str, default='data', help='name of dataset')
    parser.add_argument('--linker_design', action='store_true', default=False, help='linker task')
    parser.add_argument('--r_design', action='store_true', default=False, help='R-group task')

    parser.add_argument('--max_atoms', type=int, default=89, help='maximum atoms of generated mol')
    parser.add_argument('--atom_types', type=int, default=14, help='number of atom types in generated mol')
    parser.add_argument('--edge_types', type=int, default=4, help='number of edge types in generated mol')

    args = parser.parse_args()
    assert (args.linker_design and not args.r_design) or (args.r_design and not args.linker_design), 'please specify either linker design or R-group design'
    
    print('start loading data...')
    with open(args.data, 'r') as f:
        lines = f.readlines()
    data=[]
    for line in lines:
        toks = line.strip().split(' ')
        smi_mol, smi_generate, smi_frags = toks
        data.append({'smi_mol': smi_mol, 'smi_generate': smi_generate, 
                        'smi_frags': smi_frags})

    print('start preprocessing...')
    if args.linker_design:
        processed_data = preprocess_linker(data)

    else:
        processed_data = preprocess_r(data)

    if  not os.path.exists(args.save_fold):
        os.makedirs(args.save_fold)
    with open(args.save_fold + args.name + '.json', 'w') as f:
        json.dump(processed_data, f)
    
    # convert the processed data to Adjacency Matrix 
    full_nodes = []
    full_edges = []
    frag_nodes=[]
    frag_edges=[]
    gen_len = []
    v_to_keep=[]
    exit_point=[]
    full_smi=[]
    frag=[]
    for line in processed_data:
        # output molecules
        full_node = torch.zeros([args.max_atoms, args.atom_types]) # (89, 14)
        for i in range(len(line['node_features_out'])):
            for j in range(len(line['node_features_out'][0])):
                full_node[i][j] = line['node_features_out'][i][j]
        full_nodes.append(full_node)
        full_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms]) # (4, 89, 89)
        for i in (line['graph_out']):
            start=i[0]
            end=i[2]
            edge=i[1]
            full_edge[edge,start,end]=1.0
            full_edge[edge,end,start]=1.0
        full_edges.append(full_edge)
        gen_len.append(len(line['node_features_out'])-len(line['node_features_in']))

        # input fragments
        frag_node = torch.zeros([args.max_atoms, args.atom_types]) # (89, 14)
        for i in range(len(line['node_features_in'])):
            for j in range(len(line['node_features_in'][0])):
                frag_node[i][j]=line['node_features_in'][i][j]

        frag_nodes.append(frag_node)
        frag_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms]) # (4, 89, 89)
        for i in (line['graph_in']):
            start=i[0]
            end=i[2]
            edge=i[1]
            frag_edge[edge,start,end]=1.0
            frag_edge[edge,end,start]=1.0
        frag_edges.append(frag_edge)

        v_to_keep.append(line['v_to_keep'][-1])
        exit_point.append(line['exit_points'])
        full_smi.append(line['smiles_out'])
        frag.append(line['smiles_in'])
    
    full_nodes=torch.tensor([item.detach().numpy() for item in full_nodes])
    full_edges=torch.tensor([item.detach().numpy() for item in full_edges])
    frag_nodes=torch.tensor([item.detach().numpy() for item in frag_nodes])
    frag_edges=torch.tensor([item.detach().numpy() for item in frag_edges])

    np.save(args.save_fold + 'full_nodes', full_nodes)
    np.save(args.save_fold + 'full_edges', full_edges)
    np.save(args.save_fold + 'frag_nodes', frag_nodes)
    np.save(args.save_fold + 'frag_edges', frag_edges)
    np.save(args.save_fold + 'gen_len', gen_len)
    np.save(args.save_fold + 'v_to_keep', v_to_keep)
    np.save(args.save_fold + 'exit_point', exit_point)
    np.save(args.save_fold + 'full_smi', full_smi)
    np.save(args.save_fold + 'frag', frag)

    fp = open(args.save_fold + 'config.txt', 'w')
    config = dict()
    config['atom_list'] = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}
    config['node_dim'] = args.atom_types
    config['max_size'] = args.max_atoms
    config['bond_dim'] = args.edge_types
    fp.write(str(config))
    fp.close()

    print('done!')
