import numpy as np
import re
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import rdMolAlign
from joblib import Parallel, delayed
import sascorer
from rdkit.Chem.QED import qed
from utils import *
import rdkit_conf_parallel
from tqdm import *
import environment as env

def check_atoms(smi):
    flag = True
    mol = Chem.MolFromSmiles(smi)
    for bond in mol.GetBonds(): 
        if bond.IsInRing() == False and mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() != 'C' and mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() != 'C':
            return False

    for x in mol.GetRingInfo().AtomRings():
        if len(x) == 3 or len(x) == 4:
            for atom in x:
                if str(mol.GetAtomWithIdx(atom).GetHybridization()) != 'SP3':
                    return False
    return flag

parser = argparse.ArgumentParser(description='FFLOM model')
parser.add_argument('--train_data', type=str, help='path of training set', required=True)
parser.add_argument('--gen_data', type=str, help='path of generated molecules', required=True)
parser.add_argument('--n_cores', type=int, default=1, help='cores to use')
parser.add_argument('--linker_design', action='store_true', default=False, help='linker task')
parser.add_argument('--r_design', action='store_true', default=False, help='R-group task')
parser.add_argument('--ref_path', type=str, default='zinc_250k_valid_test_only.sdf', help='path of 3D conformations of ground truth molecules')
args = parser.parse_args()
assert (args.linker_design and not args.r_design) or (args.r_design and not args.linker_design), 'please specify either linker design or R-group design'
    
# Load molecules
# FORMAT: (Starting fragments (SMILES), Original molecule (SMILES), Generated molecule (SMILES))
with open(args.gen_data, 'r') as f:
    full = [line.strip('\n') for line in f]

frag_mols = [line.split()[0] for line in full]
in_mols = [line.split()[1] for line in full]
gen_mols = [line.split()[2] for line in full]
clean_frags = Parallel(n_jobs=args.n_cores)(delayed(remove_dummys)(smi) for smi in frag_mols)

# ---------------------------------
# 1. Check valid
results = []
for in_mol, frag_mol, gen_mol, clean_frag in zip(in_mols, frag_mols, gen_mols, clean_frags):
    try:
        if len(Chem.MolFromSmiles(gen_mol).GetSubstructMatch(Chem.MolFromSmiles(clean_frag)))>0:
            results.append([in_mol, frag_mol, gen_mol, clean_frag])#
    except:
        continue
print("Number of generated SMILES: \t%d" % len(full))
print("Number of valid SMILES: \t%d" % len(results))
print("%% Valid: \t\t\t%.2f%%" % (len(results)/len(full)*100))

if args.linker_design:
    linkers = []
    for m in results:
        linkers.append(get_linker(Chem.MolFromSmiles(m[2]), Chem.MolFromSmiles(m[3]), m[1]) ) 
    for i, linker in enumerate(linkers): # Standardise linkers
        if linker == "":
            continue
        try:
            linker_canon = Chem.MolFromSmiles(re.sub('[0-9]+\*', '*', linker))
            Chem.rdmolops.RemoveStereochemistry(linker_canon)
            linkers[i] = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker_canon))
        except:
            continue
    for i in range(len(results)):
        results[i].append(linkers[i])

    # filter
    results_filt=[]
    for res in results:
        if check_atoms(res[4])==True:
            results_filt.append(res)
    results=results_filt

    # 2. check uniqueness
    results_dict = {}
    for res in results:
        if res[0]+'.'+res[1] in results_dict: # Unique identifier - starting fragments and original molecule
            results_dict[res[0]+'.'+res[1]].append(tuple(res))
        else:
            results_dict[res[0]+'.'+res[1]] = [tuple(res)]
    print("Unique molecules: %.2f%%" % (unique(results_dict.values())*100))

    # 3. check novelty
    linkers_train = []
    with open(args.train_data, 'r') as f:
        for line in f:
            toks = line.strip().split(' ')
            linkers_train.append(toks[1])
    linkers_train_nostereo = []
    for smi in list(set(linkers_train)):
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        linkers_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        
    linkers_train_nostereo = {smi.replace(':1', '').replace(':2', '') for smi in set(linkers_train_nostereo)} # Standardise / canonicalise training set linkers
    linkers_train_canon = []
    for smi in list(linkers_train_nostereo):
        try:
            linkers_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))
        except:
            #print('error')
            continue

    linkers_train_canon_unique = list(set(linkers_train_canon))
    print("Number of unique linkers: %d" % len(linkers_train_canon_unique))
    # Check novelty of generated molecules
    count_novel = 0
    for res in results:
        if res[4] in linkers_train_canon_unique:
            continue
        else:
            count_novel +=1
    print("Novel linkers: %.2f%%" % (count_novel/len(results)*100))

else: # R-group case
    rs = []
    for m in results:
        rs.append(get_r(Chem.MolFromSmiles(m[2]), Chem.MolFromSmiles(m[3]), m[1])  )
    for i, r in enumerate(rs):
        if r == "":
            continue
        try:
            r_canon = Chem.MolFromSmiles(re.sub('[0-9]+\*', '*', r))
            Chem.rdmolops.RemoveStereochemistry(r_canon)
            rs[i] = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(r_canon))
        except:
            #print('error')
            continue
    for i in range(len(results)):
        results[i].append(rs[i])

    # filter
    results_filt=[]
    for res in results:
        if check_atoms(res[4])==True:
            results_filt.append(res)
    results=results_filt

    # 2. check uniqueness
    results_dict = {}
    for res in results:
        if res[0]+'.'+res[1] in results_dict: # Unique identifier - starting fragments and original molecule
            results_dict[res[0]+'.'+res[1]].append(tuple(res))
        else:
            results_dict[res[0]+'.'+res[1]] = [tuple(res)]
    print("Unique molecules: %.2f%%" % (unique(results_dict.values())*100))

    # 3. check novelty
    rs_train = []
    with open(args.train_data, 'r') as f:
        for line in f:
            toks = line.strip().split(' ')
            rs_train.append(toks[1])

    rs_train_nostereo = []
    for smi in list(set(rs_train)):
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        rs_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))

    rs_train_nostereo = {smi.replace(':1', '') for smi in set(rs_train_nostereo)}
    rs_train_canon = []
    for smi in list(rs_train_nostereo):
        try:
            rs_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))
        except:
            #print('error')
            continue

    rs_train_canon_unique = list(set(rs_train_canon))
    print("Number of unique rs: %d" % len(rs_train_canon_unique))
    count_novel = 0
    for res in results:
        if res[4] in rs_train_canon_unique:
            continue
        else:
            count_novel +=1
    print("Novel rs: %.2f%%" % (count_novel/len(results)*100))

# 4. check recovery 
# run recon.py (for linker case) or recon_r.py (for r-group case)

# 5. check SA/QED/plogP
smi=[line[2] for line in results]
mol=[Chem.MolFromSmiles(s) for s in smi]
sa=[]
qe=[]
plogp=[]
for m in mol:
    sa.append(sascorer.calculateScore(m))
    qe.append(qed(m))
    plogp.append(env.penalized_logp(m))    
print('SA score: %.3f ; QED: %.3f ; plogP: %.3f\n'%(sum(sa)/len(mol), sum(qe)/len(mol), sum(plogp)/len(mol)))

print('done!')