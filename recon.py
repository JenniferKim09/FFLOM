'''
python recon.py --path ./data_preprocessed/zinc_test_linker/ --gen --gen_out_path ./ --batch_size 32 --lr 0.001 --epochs 10 --shuffle --name 0904 --num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 --seed 66666666 --init_checkpoint ./good_ckpt/checkpoint306 --gen_num 100 --edge_unroll 12
'''

import torch
from time import time
import argparse
import numpy as np
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

parser = argparse.ArgumentParser(description='FFLOM model')

# ******data args******
parser.add_argument('--dataset', type=str, default='zinc250k', help='dataset')
parser.add_argument('--path', type=str, help='path of dataset', required=True)


parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')

# ******model args******
parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
parser.add_argument('--deq_coeff', type=float, default=0.9, help='dequantization coefficient.(only for deq_type random)')
parser.add_argument('--num_flow_layer', type=int, default=6, help='num of affine transformation layer in each timestep')
parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
#TODO: Disentangle num of hidden units for gcn layer, st net layer.
parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

parser.add_argument('--st_type', type=str, default='exp', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

# ******for sigmoid st net only ******
parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

# ******for exp st net only ******

# ******for softplus st net only ******

# ******optimization args******
parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
parser.add_argument('--train', action='store_true', default=False, help='do training.')
parser.add_argument('--save', action='store_true', default=True, help='Save model.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--is_bn', action='store_true', default=True, help='batch norm on node embeddings.')
parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
parser.add_argument('--divide_loss', action='store_true', default=True, help='divide loss by length of latent.')
parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

parser.add_argument('--show_loss_step', type=int, default=100)

# ******generation args******
parser.add_argument('--temperature', type=float, default=0.7, help='temperature for normal distribution')
parser.add_argument('--min_atoms', type=int, default=10, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
parser.add_argument('--max_atoms', type=int, default=88, help='maximum #atoms of generated mol')    
parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
parser.add_argument('--gen', action='store_true', default=False, help='generate')
parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)

from train import Trainer,save_model,read_molecules
full_nodes, full_edges, frag_nodes,frag_edges, linker_len, v_to_keep, exit_point, full_smi, data_config = read_molecules(args.path)
train_dataloader = DataLoader(CondDataset(full_nodes, full_edges, frag_nodes,frag_edges,linker_len, exit_point, v_to_keep, full_smi),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=full_smi)

checkpoint = torch.load(args.init_checkpoint)
trainer._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
trainer._model.eval()

temperature=args.temperature
#mute=True
#max_atoms=trainer.args.max_atoms

with torch.no_grad():
    num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
    num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
    num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 
                            7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}
    atom_types = {0:'Br1(0)', 1:'C4(0)', 2:'Cl1(0)', 3:'F1(0)', 4:'H1(0)', 5:'I1(0)',
            6:'N2(-1)', 7:'N3(0)', 8:'N4(1)', 9:'O1(-1)', 10:'O2(0)', 11:'S2(0)', 12:'S4(0)', 13:'S6(0)'}                        
    maximum_valence = {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3}
    prior_node_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.node_dim]).cuda(), 
                                        temperature * torch.ones([trainer._model.node_dim]).cuda())
    prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.bond_dim]).cuda(), 
                                        temperature * torch.ones([trainer._model.bond_dim]).cuda())

def reconstruct(num, x_deq, adj_deq, noise=0):
    node_symbol=[]
    for i in frag_nodes[num]:
        if max(i)>0:
            node_symbol.append(np.argmax(i))
    # initial frag input
    rw_mol = Chem.RWMol()
    valences = [maximum_valence[s] for s in node_symbol]
    #add atoms
    for number in node_symbol:
        new_atom = Chem.Atom(num2symbol[number])
        charge_num=int(atom_types[number].split('(')[1].strip(')'))
        new_atom.SetFormalCharge(charge_num)
        rw_mol.AddAtom(new_atom)
    #add edges
    for bond in range(3): #(4,89,89)
        for start in range(89):
            for end in range(start+1,89):
                if frag_edges[num][bond][start][end]==1:
                    rw_mol.AddBond(start, end, num2bond[bond])

    # update valences
    for i in range(len(node_symbol)):
        if i not in exit_point[num]:
            valences[i] = 0
        else:
            valences[i] = 1
    cur_node_features = torch.tensor(frag_nodes[num]).unsqueeze(0).cuda() #(1,89,14)
    cur_adj_features = torch.tensor(frag_edges[num]).unsqueeze(0).cuda() #(1,4,89,89)

    new_nodes=[i for i in range(len(node_symbol),(len(node_symbol)+linker_len[num]))] 
    
    #start generating
    for i,new_node in enumerate(new_nodes): #[20,21,22,23]
        max_x=0
        latent_node = x_deq[0][14*new_node:14*(new_node+1)].clone().view(1,-1) #(1, 14)
        latent_node += noise * prior_node_dist.sample().view(1, -1)
        latent_node = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                                latent_node, mode=0).view(-1) # (14, )
        max_x = max(latent_node)

        feature_id = torch.argmax(latent_node).item()
        cur_node_features[0, new_node, feature_id] = 1.0
        add_atoms(rw_mol, [feature_id])
        #print(new_node, num2symbol[feature_id])

        new_edges=[i for i,x in enumerate(valences) if x > 0][1:] #skip one exit point
        valences.append(maximum_valence[feature_id])
        max_y=0
        #bond_num=0
        
        for new_edge in new_edges:
            #flag=True
            #[19,20]
            edge = int(12*11/2 + (new_node-12+1)*12 - (new_node-new_edge))
            latent_edge = adj_deq[0][4*edge:4*(edge+1)].clone().view(1,-1) #(1, 4)
            latent_edge += noise * prior_edge_dist.sample().view(1, -1)
            latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[new_edge, new_node]]).long().cuda()).view(-1)
            
            for a,x in enumerate(latent_edge):
                if a >= valences[new_edge] and a!=3:
                    latent_edge[a] *= 0
            edge_discrete_id = torch.argmax(latent_edge).item()
            if max(latent_edge)<1 or edge_discrete_id==3:
                continue

            valences[new_node] -= (edge_discrete_id + 1)
            valences[new_edge] -= (edge_discrete_id + 1)

            rw_mol.AddBond(int(new_node),int(new_edge),num2bond[edge_discrete_id])
            
            #valid = env.check_valency(rw_mol)
            cur_adj_features[0, edge_discrete_id, new_node, new_edge] = 1.0
            cur_adj_features[0, edge_discrete_id, new_edge, new_node] = 1.0

    # connect exit_point
    if valences[new_node]>0:
        rw_mol.AddBond(int(min(exit_point[num])),int(new_node),num2bond[0])
    else:
        new_edges=[i for i,x in enumerate(valences) if x > 0][1:]
        flag=True
        edge_p=[]
        for new_edge in new_edges:#[19,20]
            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
            latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[new_edge, min(exit_point[num])]]).long().cuda()).view(-1)
            for a,x in enumerate(latent_edge):
                if a >= valences[new_edge] and a!=3:
                    latent_edge[a] *= 0
            edge_p.append(latent_edge)
        if edge_p==[]:
            return None
        index=torch.argmax(torch.tensor([max(e) for e in edge_p])).item()
        edge_discrete_id = torch.argmax(torch.tensor(edge_p[index])).item()
        if  edge_discrete_id==3: #no valences remain
            return None
        chosen_edge=new_edges[index]
        rw_mol.AddBond(int(min(exit_point[num])),int(chosen_edge),num2bond[edge_discrete_id])

    mol = rw_mol.GetMol()
    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
    smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
    if '.' in smiles or Chem.MolFromSmiles(smiles)==None:
        return None
    #final_mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def cano(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile),isomericSmiles=False)

smiles=[]
recon=0
for idx in range(len(frag_nodes)):
    inp_node_features = torch.tensor(full_nodes[idx]).unsqueeze(0).cuda() #(B, N, node_dim)
    inp_adj_features = torch.tensor(full_edges[idx]).unsqueeze(0).cuda() #(B, 4, N, N) 
    z,logdet,_ = trainer._model(inp_node_features, inp_adj_features)
    x_deq=z[0]
    adj_deq=z[1]
    #print('start generating...')
    smile = reconstruct(idx, x_deq, adj_deq, noise=0)
    if smile != None:
        smiles.append(smile)
        if smile==cano(full_smi[idx]):
            recon+=1
    print('%s: valid:%.5f | unique:%.5f | recon:%.5f'%(idx, len(smiles)/(idx+1), 
                                len(set(smiles))/max(len(smiles),1),recon/(idx+1)))