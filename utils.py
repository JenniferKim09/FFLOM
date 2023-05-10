# coding=utf-8
"""
Anonymous author
part of codes are taken from gcpn/graphRNN's open-source code.
Description: load raw smiles, construct node/edge matrix.
"""

import sys
import os
import csv
import numpy as np
import networkx as nx
import random

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMMPA
from rdkit.Chem import rdMolAlign
import torch
import torch.nn.functional as F
import sascorer
from itertools import product
from joblib import Parallel, delayed
#import calc_SC_RDKit

bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
atom_types = {0:'Br1(0)', 1:'C4(0)', 2:'Cl1(0)', 3:'F1(0)', 4:'H1(0)', 5:'I1(0)',
            6:'N2(-1)', 7:'N3(0)', 8:'N4(1)', 9:'O1(-1)', 10:'O2(0)', 11:'S2(0)', 12:'S4(0)', 13:'S6(0)'}
num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 
                            7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}
                            
def add_atoms(mol, node_list):
    for number in node_list:
        new_atom = Chem.Atom(num2symbol[number])
        charge_num=int(atom_types[number].split('(')[1].strip(')'))
        new_atom.SetFormalCharge(charge_num)
        mol.AddAtom(new_atom)
        
def show_idx(mol):
    starting_point_2d = Chem.Mol(mol)
    _ = AllChem.Compute2DCoords(starting_point_2d)
    return Draw.MolToImage(mol_with_atom_index(starting_point_2d))

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z

def dataset_info(dataset):  
    if dataset == 'qm9':
        return {'atom_types': ["H", "C", "N", "O", "F"],
                'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                }
    elif dataset == 'zinc':
        return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                               'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
                'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3},
                'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'},
                'bucket_sizes': np.array(
                    [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
                }

def rearrange(re_idx_pre, re_idx_pos):
    temp = []
    for i in range(len(re_idx_pre)):
        temp.append(re_idx_pre[re_idx_pos[i]])
    return temp

def align_mol_to_frags(smi_molecule, smi_linker, smi_frags):
    try:
        # Load SMILES as molecules
        mol = Chem.MolFromSmiles(smi_molecule)
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # Include dummy atoms in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
    
        # Renumber molecule based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Get matches to fragments and linker
        qfrag = Chem.AdjustQueryProperties(frags,qp)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker,qp)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))

        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            # If perfect match, break
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)

        nodes_to_keep = [i for i in range(len(frag_match))]
        
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            sub_idx.remove(dummy_idx[0])
            sub_idx.append(dummy_idx[0])
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)

        # Get exit vectors
        exit_vectors = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())

        if len(exit_vectors) != 2:
            print("Incorrect number of exit vectors")

        return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors

    except:
        print("Could not align")
        return ([],[]), [], []

def align_mol_to_frags_elaboration(smiles_mol, smiles_frag):
    #Amended function which takes a single fragment as input
    try:
        smiles_frags = smiles_frag + '.[*:2]'
        mols_to_align = [Chem.MolFromSmiles(smiles_mol), Chem.MolFromSmiles(smiles_frags)]
        frags = [Chem.MolFromSmiles(smiles_frag)]

        # Include dummy in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True

        # Renumber based on frags (incl. dummy atoms)
        aligned_mols = []
        for i, mol in enumerate(mols_to_align):
            sub_idx = []
            for frag in frags:
                # Align to frags
                qfrag = Chem.AdjustQueryProperties(frag,qp)
                sub_idx += list(mol.GetSubstructMatch(qfrag))
            nodes_to_keep = [i for i in range(len(sub_idx))]
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))

        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    neighbor=nei.GetIdx()
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            sub_idx.remove(neighbor)
            sub_idx.append(neighbor)
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)

        # Get exit vectors
        exit_vectors = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())

        if len(exit_vectors) != 1:
            print("Incorrect number of exit vectors")

        return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors

    except:
        print("Could not align")
        return ([],[]), [], []

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False
def to_graph_mol(mol, dataset):
    if mol is None:
        return [], []
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_idx, end_idx = min(begin_idx, end_idx), max(begin_idx, end_idx)
        if mol.GetAtomWithIdx(begin_idx).GetAtomicNum() == 0 or mol.GetAtomWithIdx(end_idx).GetAtomicNum() == 0:
            continue
        else:
            edges.append((begin_idx, bond_dict[str(bond.GetBondType())], end_idx))
            assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        if dataset=='qm9' or dataset=="cep":
            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types'])))
        elif dataset=='zinc': # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)

            if atom_str not in dataset_info(dataset)['atom_types']:
                if "*" in atom_str:
                    continue
                else:
                    # print('unrecognized atom type %s' % atom_str)
                    return [], []

            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))

    return nodes, edges

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

    
def get_maxlen_of_bfs_queue(path):
    """
    Calculate the maxlen of bfs queue.
    """
    fp = open(path, 'r')
    max_all = []
    cnt = 0
    for smiles in fp:
        cnt += 1
        if cnt % 10000 == 0:
            print('cur cnt %d' % cnt)
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        #adj = construct_adj_matrix(mol)
        graph = mol_to_nx(mol)
        N = len(graph.nodes)
        for i in range(N):
            start = i
            order, max_ = bfs_seq(graph, start)
            max_all.append(max_)
    print(max(max_all))


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        
    print('set seed for random numpy and torch')


def save_one_mol(path, old, smile, cur_iter=None, score=None):
    """
    save one molecule
    mode: append
    """
    cur_iter = str(cur_iter)

    fp = open(path, 'a')
    fp.write('%s  %s -> %s  %s\n' % (cur_iter, old, smile, str(score)))
    fp.close()


def save_one_reward(path, reward, score, loss, cur_iter):
    """
    save one iter reward/score
    mode: append
    """
    fp = open(path, 'a')
    fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp.close()

def save_one_optimized_molecule(path, org_smile, optim_smile, optim_plogp, cur_iter, ranges, sim):
    """
    path: save path
    org_smile: molecule to be optimized
    org_plogp: original plogp
    optim_smile: with shape of (4, ), containing optimized smiles with similarity constrained 0(0.2/0.4/0.6) 
    optim_plogp:  corespongding plogp 
    cur_iter: molecule index

    """
    start = ranges[0]
    end = ranges[1]
    fp1 = open(path + '/sim0_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp2 = open(path + '/sim2_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp3 = open(path + '/sim4_%d_%d' % (ranges[0], ranges[1]), 'a')
    fp4 = open(path + '/sim6_%d_%d' % (ranges[0], ranges[1]), 'a')
    out_string1 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[0], optim_plogp[0], sim[0])
    out_string2 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[1], optim_plogp[1], sim[1])
    out_string3 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[2], optim_plogp[2], sim[2])
    out_string4 = '%d|%s||%s|%.5f|%.5f\n' % (cur_iter, org_smile, optim_smile[3], optim_plogp[3], sim[3])

    fp1.write(out_string1)
    fp2.write(out_string2)
    fp3.write(out_string3)
    fp4.write(out_string4)
    #fp.write('cur_iter: %d | reward: %.5f | score: %.5f | loss: %.5f\n' % (cur_iter, reward, score, loss))
    fp1.close()
    fp2.close()
    fp3.close()
    fp4.close()


def update_optim_dict(optim_dict, org_smile, cur_smile, imp, sim):
    if imp <= 0. or sim == 1.0:
        return optim_dict
    
    else:
        if org_smile not in optim_dict:
            optim_dict[org_smile] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        if sim >= 0.:
            if imp > optim_dict[org_smile][0][1]:
                optim_dict[org_smile][0][0] = cur_smile
                optim_dict[org_smile][0][1] = imp
                optim_dict[org_smile][0][2] = sim

        if sim >= 0.2:
            if imp > optim_dict[org_smile][1][1]:
                optim_dict[org_smile][1][0] = cur_smile
                optim_dict[org_smile][1][1] = imp
                optim_dict[org_smile][1][2] = sim

        if sim >= 0.4:
            if imp > optim_dict[org_smile][2][1]:
                optim_dict[org_smile][2][0] = cur_smile
                optim_dict[org_smile][2][1] = imp
                optim_dict[org_smile][2][2] = sim

        if sim >= 0.6:
            if imp > optim_dict[org_smile][3][1]:
                optim_dict[org_smile][3][0] = cur_smile
                optim_dict[org_smile][3][1] = imp
                optim_dict[org_smile][3][2] = sim  
        return optim_dict                          


def update_total_optim_dict(total_optim_dict, optim_dict):
    all_keys = list(optim_dict.keys())
    for key in all_keys:
        if key not in total_optim_dict:
            total_optim_dict[key] = [['', -100, -1], ['', -100, -1], ['', -100, -1], ['', -100, -1]]
        
        if optim_dict[key][0][1] > total_optim_dict[key][0][1]:
            total_optim_dict[key][0][0] = optim_dict[key][0][0]
            total_optim_dict[key][0][1] = optim_dict[key][0][1]
            total_optim_dict[key][0][2] = optim_dict[key][0][2]

        if optim_dict[key][1][1] > total_optim_dict[key][1][1]:
            total_optim_dict[key][1][0] = optim_dict[key][1][0]
            total_optim_dict[key][1][1] = optim_dict[key][1][1]
            total_optim_dict[key][1][2] = optim_dict[key][1][2]

        if optim_dict[key][2][1] > total_optim_dict[key][2][1]:
            total_optim_dict[key][2][0] = optim_dict[key][2][0]
            total_optim_dict[key][2][1] = optim_dict[key][2][1]
            total_optim_dict[key][2][2] = optim_dict[key][2][2]

        if optim_dict[key][3][1] > total_optim_dict[key][3][1]:
            total_optim_dict[key][3][0] = optim_dict[key][3][0]
            total_optim_dict[key][3][1] = optim_dict[key][3][1]
            total_optim_dict[key][3][2] = optim_dict[key][3][2]
    return total_optim_dict                                    

def add_edge_mat(amat, src, dest, e, considering_edge_type=True):
    if considering_edge_type:
        amat[e, dest, src] = 1
        amat[e, src, dest] = 1
    else:
        amat[src, dest] = 1
        amat[dest, src] = 1 

def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True, considering_edge_type=True):
    if considering_edge_type:
        amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e)
    else:
        amat = np.zeros((max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e, considering_edge_type=False)
    return amat

def unique(results):
    total_dupes = 0
    total = 0
    for res in results:
        original_num = len(res)
        test_data = set(res)
        new_num = len(test_data)
        total_dupes += original_num - new_num
        total += original_num   
    return 1 - total_dupes/float(total)

def check_recovered_original_mol(results):
    outcomes = []
    for res in results:
        success = False
        # Load original mol and canonicalise
        orig_mol = Chem.MolFromSmiles(res[0][0])
        Chem.RemoveStereochemistry(orig_mol)
        orig_mol = Chem.MolToSmiles(Chem.RemoveHs(orig_mol))
        #orig_mol = MolStandardize.canonicalize_tautomer_smiles(orig_mol)
        # Check generated mols
        for m in res:
            gen_mol = Chem.MolFromSmiles(m[2])
            Chem.RemoveStereochemistry(gen_mol)
            gen_mol = Chem.MolToSmiles(Chem.RemoveHs(gen_mol))
            #gen_mol = MolStandardize.canonicalize_tautomer_smiles(gen_mol)
            if gen_mol == orig_mol:
                outcomes.append(True)
                success = True
                break
        if not success:
            outcomes.append(False)
    return outcomes

def calc_sa_score_mol(mol, verbose=False):
    if mol is None:
        if verbose:
            print("Error passing: %s" % smi)
        return None
    # Synthetic accessibility score
    return sascorer.calculateScore(mol)

def check_ring_filter(linker):
    check = True 
    
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    '''
    for bond in linker.GetBonds(): 
        if bond.IsInRing() == False and linker.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() != 'C' and linker.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() != 'C':
            check = False

    for x in linker.GetRingInfo().AtomRings():
        if len(x) == 3 or len(x) == 4:
            for atom in x:
                if str(linker.GetAtomWithIdx(atom).GetHybridization()) != 'SP3':
                    check = False
                    #print('3 or 4 ring with double bonds')
    '''
    return check

def check_pains(mol, pains_smarts):
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True


def check_2d_filters(toks, pains_smarts, count=0, verbose=False, design_task="linker"):
    # Progress
    if verbose:
        if count % 1000 == 0:
            print("\rProcessed: %d" % count, end = '')
    
    # Input format: (Full Molecule (SMILES), Linker (SMILES), Unlinked Fragment 1 (SMILES), Unlinked Fragment 2 (SMILES))
    if design_task == "linker":
        frags = Chem.MolFromSmiles(toks[2] + '.' + toks[3])
    else:
        frags = Chem.MolFromSmiles(toks[2])
    linker = Chem.MolFromSmiles(toks[1])
    full_mol = Chem.MolFromSmiles(toks[0])
    # Remove dummy atoms from unlinked fragments
    du = Chem.MolFromSmiles('*')
    clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frags,du,Chem.MolFromSmiles('[H]'),True)[0])
    
    # Check: Unlinked fragments in full molecule
    if len(full_mol.GetSubstructMatch(clean_frag))>0:
        # Check: SA score improved from unlinked fragments to full molecule
        if calc_sa_score_mol(full_mol) < calc_sa_score_mol(frags):
            # Check: No non-aromatic rings with double bonds
            if check_ring_filter(linker): 
                # Check: Pass pains filters
                if check_pains(full_mol, pains_smarts):
                    return True
            else:
                if check_ring_filter(linker):
                    print(toks)

    return False

def check_2d_filters_dataset(fragmentations, n_cores=1, pains_smarts_loc='./wehi_pains.csv', design_task="linker"):
    # Load pains filters
    with open(pains_smarts_loc, 'r') as f:
        pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]
        
    with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
        results = parallel(delayed(check_2d_filters)(toks, pains_smarts, count, True, design_task=design_task) for count, toks in enumerate(fragmentations))

    fragmentations_filtered = [toks for toks, res in zip(fragmentations, results) if res]
    
    return fragmentations_filtered

def calc_2d_filters(toks, pains_smarts): 
    try:
        # Input format: (Full Molecule (SMILES), Linker (SMILES), Unlinked Fragments (SMILES))
        frags = Chem.MolFromSmiles(toks[2])
        linker = Chem.MolFromSmiles(toks[1])
        full_mol = Chem.MolFromSmiles(toks[0])
        # Remove dummy atoms from unlinked fragments
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frags,du,Chem.MolFromSmiles('[H]'),True)[0])
    
        res = []
        # Check: Unlinked fragments in full molecule
        if len(full_mol.GetSubstructMatch(clean_frag))>0:
            # Check: SA score improved from unlinked fragments to full molecule
            if calc_sa_score_mol(full_mol) < calc_sa_score_mol(frags):
                res.append(True)
            else:
               res.append(False)
            # Check: No non-aromatic rings with double bonds
            if check_ring_filter(linker): 
               res.append(True)
            else:
                res.append(False)
            # Check: Pass pains filters
            if check_pains(full_mol, pains_smarts):
               res.append(True)
            else:
               res.append(False)     
        return res
    except:
        return [False, False, False]

def calc_filters_2d_dataset(results, pains_smarts_loc, n_cores=1):
    # Load pains filters
    with open(pains_smarts_loc, 'r') as f:
        pains_smarts = [Chem.MolFromSmarts(line[0], mergeHs=True) for line in csv.reader(f)]
        
    with Parallel(n_jobs=n_cores, backend='multiprocessing') as parallel:
        filters_2d = parallel(delayed(calc_2d_filters)([toks[2], toks[4], toks[1]], pains_smarts) for toks in results)
        
    return filters_2d
# fragments
def remove_dummys(smi_string):
    return Chem.MolToSmiles(Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(smi_string),Chem.MolFromSmiles('*'),Chem.MolFromSmiles('[H]'),True)[0]))

def get_linker(full_mol, clean_frag, starting_point):
    # INPUT FORMAT: molecule (RDKit mol object), clean fragments (RDKit mol object), starting fragments (SMILES)
        
    # Get matches of fragments
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    
    # If no matches, terminate
    if len(matches) == 0:
        print("No matches")
        return ""

    # Get number of atoms in linker
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if linker_len == 0:
        return ""
    
    # Setup
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    
    poss_linker = []

    if len(matches)>0:
        # Loop over matches
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            # Get linker atoms
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            linker_bonds = []
            atoms_joined_to_linker = []
            # Loop over starting fragments atoms
            # Get (i) bonds between starting fragments and linker, (ii) atoms joined to linker
            for idx_to_delete in sorted(match, reverse=True):
                nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
                intersect = set(nei).intersection(set(linker_atoms))
                if len(intersect) == 1:
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,list(intersect)[0]).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
                elif len(intersect) > 1:
                    for idx_nei in list(intersect):
                        linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,idx_nei).GetIdx())
                        atoms_joined_to_linker.append(idx_to_delete)
                        
            # Check number of atoms joined to linker
            # If not == 2, check next match
            if len(set(atoms_joined_to_linker)) != 2:
                continue
            
            # Delete starting fragments atoms
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            
            linker = Chem.Mol(mol_rw)
            # Check linker required num atoms
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                # Delete linker atoms
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                # Check there are two disconnected fragments
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    # Fragment molecule into starting fragments and linker
                    fragmented_mol = Chem.FragmentOnBonds(mol_to_break, linker_bonds)
                    # Remove starting fragments from fragmentation
                    linker_to_return = Chem.Mol(fragmented_mol)
                    qp = Chem.AdjustQueryParameters()
                    qp.makeDummiesQueries=True
                    for f in starting_point.split('.'):
                        qfrag = Chem.AdjustQueryProperties(Chem.MolFromSmiles(f),qp)
                        linker_to_return = AllChem.DeleteSubstructs(linker_to_return, qfrag, onlyFrags=True)
                    
                    # Check linker is connected and two bonds to outside molecule
                    if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 2:
                        Chem.Kekulize(linker_to_return, clearAromaticFlags=True)
                        # If for some reason a starting fragment isn't removed (and it's larger than the linker), remove (happens v. occassionally)
                        if len(Chem.rdmolops.GetMolFrags(linker_to_return)) > 1:
                            for frag in Chem.MolToSmiles(linker_to_return).split('.'):
                                if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == linker_len:
                                    return frag
                        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(linker_to_return)))
                    
                    # If not, add to possible linkers (above doesn't capture some complex cases)
                    else:
                        fragmented_mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragmented_mol), sanitize=False)
                        linker_to_return = AllChem.DeleteSubstructs(fragmented_mol, Chem.MolFromSmiles(starting_point))
                        poss_linker.append(Chem.MolToSmiles(linker_to_return))
    
    # If only one possibility, return linker
    if len(poss_linker) == 1:
        return poss_linker[0]
    # If no possibilities, process failed
    elif len(poss_linker) == 0:
        print("FAIL:", Chem.MolToSmiles(full_mol), Chem.MolToSmiles(clean_frag), starting_point)
        return ""
    # If multiple possibilities, process probably failed
    else:
        print("More than one poss linker. ", poss_linker)
        return poss_linker[0]

def get_r(full_mol, clean_frag, starting_point):
    # INPUT FORMAT: molecule (RDKit mol object), clean fragments (RDKit mol object), starting fragments (SMILES)
        
    # Get matches of fragments
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    
    # If no matches, terminate
    if len(matches) == 0:
        print("No matches")
        return ""

    # Get number of atoms in linker
    r_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if r_len == 0:
        return ""
    
    # Setup
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    
    poss_linker = []

    if len(matches)>0:
        # Loop over matches
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            # Get linker atoms
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            linker_bonds = []
            atoms_joined_to_linker = []
            # Loop over starting fragments atoms
            # Get (i) bonds between starting fragments and linker, (ii) atoms joined to linker
            for idx_to_delete in sorted(match, reverse=True):
                nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
                intersect = set(nei).intersection(set(linker_atoms))
                if len(intersect) == 1:
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,list(intersect)[0]).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
                elif len(intersect) > 1:
                    for idx_nei in list(intersect):
                        linker_bonds.append(mol_rw.GetBondBetweenAtoms(idx_to_delete,idx_nei).GetIdx())
                        atoms_joined_to_linker.append(idx_to_delete)
                        
            # Check number of atoms joined to linker
            # If not == 2, check next match
            if len(set(atoms_joined_to_linker)) != 1:
                continue
            
            # Delete starting fragments atoms
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            
            linker = Chem.Mol(mol_rw)
            # Check linker required num atoms
            if linker.GetNumHeavyAtoms() == r_len:
                mol_rw = Chem.RWMol(full_mol)
                # Delete linker atoms
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                # Check there are two disconnected fragments
                if len(Chem.rdmolops.GetMolFrags(frags)) == 1:
                    # Fragment molecule into starting fragments and linker
                    fragmented_mol = Chem.FragmentOnBonds(mol_to_break, linker_bonds)
                    # Remove starting fragments from fragmentation
                    linker_to_return = Chem.Mol(fragmented_mol)
                    qp = Chem.AdjustQueryParameters()
                    qp.makeDummiesQueries=True
                    f=starting_point
                    qfrag = Chem.AdjustQueryProperties(Chem.MolFromSmiles(f),qp)
                    linker_to_return = AllChem.DeleteSubstructs(linker_to_return, qfrag, onlyFrags=True)
                
                    # Check linker is connected and two bonds to outside molecule
                    if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 1:
                        Chem.Kekulize(linker_to_return, clearAromaticFlags=True)
                        # If for some reason a starting fragment isn't removed (and it's larger than the linker), remove (happens v. occassionally)
                        if len(Chem.rdmolops.GetMolFrags(linker_to_return)) > 1:
                            for frag in Chem.MolToSmiles(linker_to_return).split('.'):
                                if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == r_len:
                                    return frag
                        return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(linker_to_return)))
                    
                    # If not, add to possible linkers (above doesn't capture some complex cases)
                    else:
                        fragmented_mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragmented_mol), sanitize=False)
                        linker_to_return = AllChem.DeleteSubstructs(fragmented_mol, Chem.MolFromSmiles(starting_point))
                        poss_linker.append(Chem.MolToSmiles(linker_to_return))
    
    # If only one possibility, return linker
    if len(poss_linker) == 1:
        return poss_linker[0]
    # If no possibilities, process failed
    elif len(poss_linker) == 0:
        print("FAIL:", Chem.MolToSmiles(full_mol), Chem.MolToSmiles(clean_frag), starting_point)
        return ""
    # If multiple possibilities, process probably failed
    else:
        print("More than one poss linker. ", poss_linker)
        return poss_linker[0]

def get_frags(full_mol, clean_frag, starting_point):
    new=Chem.MolFromSmiles(Chem.MolToSmiles(full_mol))
    matches = list(new.GetSubstructMatches(clean_frag))
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()

    if linker_len == 0:
        return full_mol

    Chem.Kekulize(full_mol, clearAromaticFlags=True)

    all_frags = []
    all_frags_lengths = []

    if len(matches)>0:
        for match in matches:
            mol_rw = Chem.RWMol(full_mol)
            linker_atoms = list(set(list(range(full_mol.GetNumHeavyAtoms()))).difference(match))
            for idx_to_delete in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw = Chem.RWMol(full_mol)
                for idx_to_delete in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx_to_delete)
                frags = Chem.Mol(mol_rw)
                all_frags.append(frags)
                all_frags_lengths.append(len(Chem.rdmolops.GetMolFrags(frags)))
                if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                    return frags

    return all_frags[np.argmax(all_frags_lengths)]

def SC_RDKit_frag_mol(gen_mol, ref_mol, start_pt):
    try:
        # Delete linker - Gen mol
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])

        fragmented_mol = get_frags(gen_mol, clean_frag, start_pt)
        if fragmented_mol is not None:
            # Delete linker - Ref mol
            #ref_pt=ref_sdfs[0].GetProp("_StartingPoint")
            clean_frag_ref = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])
            fragmented_mol_ref = get_frags(ref_mol, clean_frag_ref, start_pt)
            if fragmented_mol_ref is not None:
                # Sanitize
                Chem.SanitizeMol(fragmented_mol)
                Chem.SanitizeMol(fragmented_mol_ref)
                # Align
                pyO3A = rdMolAlign.GetO3A(fragmented_mol, fragmented_mol_ref).Align()
                # Calc SC_RDKit score
                score = calc_SC_RDKit.calc_SC_RDKit_score(fragmented_mol, fragmented_mol_ref)
                return score
    except:
        return -0.5 # Dummy score

def SC_RDKit_frag_scores(gen_mols):
    return [SC_RDKit_frag_mol(gen_mol, ref_mol, frag_smi) for (gen_mol, ref_mol, frag_smi) in gen_mols]

def rmsd_frag_mol(gen_mol, ref_mol, start_pt):
    try:
        # Delete linker - Gen mol
        du = Chem.MolFromSmiles('*')
        clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])

        fragmented_mol = get_frags(gen_mol, clean_frag, start_pt)
        if fragmented_mol is not None:
            # Delete linker - Ref mol
            clean_frag_ref = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(start_pt),du,Chem.MolFromSmiles('[H]'),True)[0])
            fragmented_mol_ref = get_frags(ref_mol, clean_frag_ref, start_pt)
            if fragmented_mol_ref is not None:
                # Sanitize
                Chem.SanitizeMol(fragmented_mol)
                Chem.SanitizeMol(fragmented_mol_ref)
                # Align
                pyO3A = rdMolAlign.GetO3A(fragmented_mol, fragmented_mol_ref).Align()
                rms = rdMolAlign.GetBestRMS(fragmented_mol, fragmented_mol_ref)
                return rms #score
    except:
        return 100 # Dummy RMSD

def rmsd_frag_scores(gen_mols):
    return [rmsd_frag_mol(gen_mol, ref_mol, start_pt) for (gen_mol, ref_mol, start_pt) in gen_mols]

def fragment_mol(smi, cid, pattern="[#6+0;!$(*=,#[!#6])]!@!=!#[*]", design_task="linker"):
    mol = Chem.MolFromSmiles(smi)

    #different cuts can give the same fragments
    #to use outlines to remove them
    outlines = set()

    if (mol == None):
        sys.stderr.write("Can't generate mol for: %s\n" % (smi))
    else:
        if design_task == "linker":
	        frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        elif design_task == "elaboration":
	        frags = rdMMPA.FragmentMol(mol, minCuts=1, maxCuts=1, maxCutBonds=100, pattern=pattern, resultsAsMols=False)
        else:
            print("Invalid choice for design_task. Must be 'linker' or 'elaboration'.") 
        for core, chains in frags:
            if design_task == "linker":
                output = '%s,%s,%s,%s' % (smi, cid, core, chains)
            elif design_task == "elaboration":
                output = '%s,%s,%s' % (smi, cid, chains)
            if (not (output in outlines)):
                outlines.add(output)
        if not outlines:
            # for molecules with no cuts, output the parent molecule itself
            outlines.add('%s,%s,,' % (smi,cid)) 

    return outlines

def fragment_dataset(smiles, linker_min=3, fragment_min=5, min_path_length=2, linker_leq_frags=True, verbose=False, design_task="linker"):
    successes = []

    for count, smi in enumerate(smiles):
        smi = smi.rstrip()
        smiles = smi
        cmpd_id = smi

        # Fragment smi
        o = fragment_mol(smiles, cmpd_id, design_task=design_task)

        # Checks if suitable fragmentation
        for l in o:
            smiles = l.replace('.',',').split(',')
            mols = [Chem.MolFromSmiles(smi) for smi in smiles[1:]]
            if design_task == "elaboration":
			    #If the chopped portion is bigger than the fragment then we need to switch them around
                if mols[2].GetNumHeavyAtoms() < mols[1].GetNumHeavyAtoms():
                    smilesNew = [smiles[0], smiles[1], smiles[3], smiles[2]]
                    mols = [Chem.MolFromSmiles(smi) for smi in smilesNew[1:]]
                    l = ','.join(smilesNew)
            add = True
            fragment_sizes = []
            for i, mol in enumerate(mols):
                # Linker
                if i == 1:
                    linker_size = mol.GetNumHeavyAtoms()
                    # Check linker at least minimum size
                    if linker_size < linker_min:
                        add = False
                        break
                    # Check path between the fragments at least minimum
                    if design_task =="linker":
                        dummy_atom_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
                        if len(dummy_atom_idxs) != 2:
                            print("Error")
                            add = False
                            break
                        else:
                            path_length = len(Chem.rdmolops.GetShortestPath(mol, dummy_atom_idxs[0], dummy_atom_idxs[1]))-2
                            if path_length < min_path_length:
                                add = False
                                break
                # Fragments
                elif i > 1:
                    fragment_sizes.append(mol.GetNumHeavyAtoms())
                    min_fragment_size = min(fragment_sizes)
                    # Check fragment at least than minimum size
                    if mol.GetNumHeavyAtoms() < fragment_min:
                        add = False
                        break
                    if linker_leq_frags:
					    # Check linker not bigger than fragments
                        if design_task=="linker":
                            if min_fragment_size < linker_size:
                                add = False
                                break
					    # Check elaboration not more than half size of core
                        elif design_task=="elaboration":
                            if min_fragment_size < linker_size*2:
                                add = False
                                break
            if add == True:
                successes.append(l)
        
        if verbose:
            # Progress
            if count % 1000 == 0:
                print("\rProcessed smiles: " + str(count), end='')
    
    # Reformat output
    fragmentations = []
    for suc in successes:
        fragmentations.append(suc.replace('.',',').split(',')[1:])
    
    return fragmentations
