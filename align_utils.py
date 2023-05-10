from rdkit import Chem
from itertools import product

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
        if smi_linker == '':
            linker_matches = [() for _ in frags_matches]
        else:
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

##Alignment function for elaboration case
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
