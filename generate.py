import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import *
from utils import *
from train import Trainer, read_molecules
from dataloader import CondDataset
import environment as env
import copy

parser = argparse.ArgumentParser(description='FFLOM model')
# ******data args******
parser.add_argument('--path', type=str, help='path of dataset', required=True)
parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')

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
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--warm_up', action='store_true', default=True, help='Add warm-up and decay to the learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--is_bn', action='store_true', default=True, help='batch norm on node embeddings.')
parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')

parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

# ******generation args******
parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
parser.add_argument('--min_atoms', type=int, default=5, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
parser.add_argument('--max_atoms', type=int, default=89, help='maximum #atoms of generated mol')    
parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')
parser.add_argument('--len_freedom_x', type=int, default=0, help='the minimum adjust of generated length')
parser.add_argument('--len_freedom_y', type=int, default=0, help='the maximum adjust of generated length')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)
# print('seed:'+str(args.seed))

full_nodes, full_edges, frag_nodes,frag_edges, gen_len, v_to_keep, exit_point, full_smi, data_config = read_molecules(args.path)
train_dataloader = DataLoader(CondDataset(full_nodes, full_edges, frag_nodes,frag_edges,gen_len, exit_point, v_to_keep, full_smi),
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=full_smi)

checkpoint = torch.load(args.init_checkpoint)
trainer._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
trainer._model.eval()
temperature = args.temperature
# mute=True
# max_atoms=trainer.args.max_atoms

def generate_one_mol(num,len_freedom=0):
    traj=[]
    error=[]
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

    cur_node_features = torch.tensor(frag_nodes[num]).unsqueeze(0).cuda() 
    cur_adj_features = torch.tensor(frag_edges[num]).unsqueeze(0).cuda() 

    new_nodes=[i for i in range(len(node_symbol),(len(node_symbol)+gen_len[num]+len_freedom))] 
    cur_node_features_save = cur_node_features.clone()
    cur_adj_features_save = cur_adj_features.clone()
    valences_save = copy.deepcopy(valences)
    #start generating
    for i,new_node in enumerate(new_nodes): #[20,21,22,23]
        flag=True
        while flag==True:
            flag=False
            max_x=0
            while max_x < 1:
                latent_node = prior_node_dist.sample().view(1, -1) #(1, 14)
                latent_node = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                                        latent_node, mode=0).view(-1) # (14, )
                max_x = max(latent_node)
            feature_id = torch.argmax(latent_node).item()
            cur_node_features[0, new_node, feature_id] = 1.0
            add_atoms(rw_mol, [feature_id])
            traj.append([new_node, num2symbol[feature_id]])
            new_edges=[i for i,x in enumerate(valences) if x > 0][1:] #skip one exit point

            valences.append(maximum_valence[feature_id])
            max_y=0
            bond_num=[]
            while len(new_edges)>0:#
                edge_p=[]
                for new_edge in new_edges:#[19,20]
                    latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                    latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                                    mode=1, edge_index=torch.Tensor([[new_edge, new_node]]).long().cuda()).view(-1)[:3]
                    for a,x in enumerate(latent_edge):
                        if a >= valences[new_edge] or a >= valences[new_node] or x<0:
                            latent_edge[a] *= 0
                    edge_p.append(latent_edge)
                max_y = max([max(e) for e in edge_p])
                #print(max_y)
                index=torch.argmax(torch.tensor([max(e) for e in edge_p])).item()
                edge_discrete_id = torch.argmax(torch.tensor(edge_p[index])).item()
                if max_y < 1 and len(bond_num) > 0 :
                    break
                
                chosen_edge=new_edges[index]
                traj.append([new_node, chosen_edge, edge_discrete_id])
                valences[new_node] -= (edge_discrete_id + 1)
                valences[chosen_edge] -= (edge_discrete_id + 1)
                new_edges.remove(chosen_edge)
                rw_mol.AddBond(int(new_node),int(chosen_edge),num2bond[edge_discrete_id])
                bond_num.append([int(new_node),int(chosen_edge)])
                #valid = env.check_valency(rw_mol)
                cur_adj_features[0, edge_discrete_id, new_node, chosen_edge] = 1.0
                cur_adj_features[0, edge_discrete_id, chosen_edge, new_node] = 1.0
            if [i for i,x in enumerate(valences) if x > 0][1:]==[] :
                flag=True
                for b in bond_num:
                    rw_mol.RemoveBond(b[0],b[1])
                rw_mol.RemoveAtom(int(new_node))
                valences = copy.deepcopy(valences_save)
                cur_node_features = cur_node_features_save.clone()
                cur_adj_features = cur_adj_features_save.clone()
            else:
                valences_save = copy.deepcopy(valences)
                cur_node_features_save = cur_node_features.clone()
                cur_adj_features_save = cur_adj_features.clone()

    # connect exit_point
    new_edges=[i for i,x in enumerate(valences) if x > 0][1:]
    #max_y = 0

    edge_p=[]
    for new_edge in new_edges:#[19,20]
        latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
        latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                        mode=1, edge_index=torch.Tensor([[new_edge, min(exit_point[num])]]).long().cuda()).view(-1)[:3]

        edge_p.append(latent_edge[0]) #only add single bond for the second exitpoint
    if edge_p==[] : #no valences remain
        error.append(0)
        return None,error
    #max_y = max([max(e) for e in edge_p])
    index=torch.argmax(torch.tensor(edge_p)).item()
    chosen_edge=new_edges[index]
    #new_edges.remove(chosen_edge)
    rw_mol.AddBond(int(min(exit_point[num])),int(chosen_edge),num2bond[0])
    traj.append([min(exit_point[num]), chosen_edge, 0])


    mol = rw_mol.GetMol()
    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
    smiles = Chem.MolToSmiles(final_mol, isomericSmiles=False)
    if '.' in smiles or Chem.MolFromSmiles(smiles)==None:
        return None,traj

    return smiles,[]


with torch.no_grad():
    num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
    num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
    num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 
                            7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}
    maximum_valence = {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3}

    frag = np.load(args.path + 'frag.npy')

    smiles=[]

    recon = 0
    novel = 0
    repeat = args.gen_num
    len_freedom = range(args.len_freedom_x, args.len_freedom_y + 1)
    print('start generating %s * %s * %s mols' % (len(frag_nodes), repeat, len(len_freedom)))#

    for idx in range(len(frag_nodes)):
        #flag = False
        for f in len_freedom:#
            print('now len freedom = '+ str(f))
            full = []
            for t in tqdm(range(repeat)):
                prior_node_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.node_dim]).cuda(), 
                                                    temperature * torch.ones([trainer._model.node_dim]).cuda())
                prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.bond_dim]).cuda(), 
                                                    temperature * torch.ones([trainer._model.bond_dim]).cuda())
                smile,_ = generate_one_mol(idx,f)
                if smile != None:
                    # print(smile)
                    smiles.append(smile)
                    full.append([frag[idx],full_smi[idx],smile]) 

              
                if (t+1) % 50 == 0:
                    print(str(t+1) + '/' + str(repeat))
                
            print('%s: valid:%.5f | unique:%.5f '%(idx+1, len(smiles) / ((idx+1)*(t+1)), 
                                        len(set(smiles)) / len(smiles)))

            with open(args.gen_out_path, 'a') as f:
                for line in full:
                    f.write('%s %s %s\n'%(line[0], line[1], line[2]))

