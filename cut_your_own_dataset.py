import argparse
from rdkit import Chem
from utils import *
import xlrd

def dataset_info(dataset): 
    if dataset=='zinc':
        return { 'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)','S4(0)', 'S6(0)'],
                 'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5:1, 6:2, 7:3, 8:4, 9:1, 10:2, 11:2, 12:4, 13:6},
                 'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5:'I', 6:'N', 7:'N', 8:'N', 9:'O', 10:'O', 11:'S', 12:'S', 13:'S'},
               }
    else:
        exit(1)

def check_smi_atom_types(smi, dataset='zinc', verbose=False):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)

        if atom_str not in dataset_info(dataset)['atom_types']:
            if "*" in atom_str:
                continue
            else:
                if verbose:
                    print('unrecognized atom type %s' % atom_str)
                return False
    return True

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description='FFLOM model')
    parser.add_argument('--xls_path', type=str, help='path of molecules', required=True)
    parser.add_argument('--output_path', type=str, default='./dataset/data.txt', help='path to save data')
    parser.add_argument('--linker_design', action='store_true', default=False, help='linker task')
    parser.add_argument('--r_design', action='store_true', default=False, help='R-group task')
    parser.add_argument('--n_cores', type=int, default=4, help='cores to use')

    parser.add_argument('--linker_min', type=int, default=5, help='minimun size of cut linker')
    parser.add_argument('--fragment_min', type=int, default=5, help='minimun size of cut fragment')
    parser.add_argument('--min_path_length', type=int, default=2, help='minimun length between two dummy atoms(in linker case)')

    args = parser.parse_args()
    assert (args.linker_design and not args.r_design) or (args.r_design and not args.linker_design), 'please specify either linker design or R-group design'
    if args.linker_design:
        design_task = 'linker'
    else:
        design_task = 'elaboration'

    
    # Load data
    xl = xlrd.open_workbook(args.xls_path) # use your xls file of molecules
    table = xl.sheets()[0]
    col = table.col_values(0,0) # (0,0) should be adjusted according to the position of data in xls
    smiles_filt = []
    errors = 0
    for i, smi in enumerate(col):
        if check_smi_atom_types(smi):
            smiles_filt.append(smi)
        else:
            errors += 1

    if i % 1000 == 0:
        print("\rProcessed smiles: %d" % i, end='')

    print("Original num entries: \t\t\t%d" % len(col))
    print("Number with permitted atom types: \t%d" % len(smiles_filt))
    print("Number of errors: \t\t\t%d" % errors)

    # Fragment dataset
    fragmentations = fragment_dataset(smiles_filt, linker_min=args.linker_min, fragment_min=args.fragment_min, min_path_length=args.min_path_length, linker_leq_frags=True, verbose=True, design_task=design_task)

    print("Processed smiles: \t%d" % len(smiles_filt))
    print("Num fragmentations: \t%d" % len(fragmentations))

    # Filter fragmentions based on 2D properties
    fragmentations_filt = check_2d_filters_dataset(fragmentations, n_cores=args.n_cores, pains_smarts_loc="wehi_pains.csv", design_task=design_task)

    print("Number fragmentations: \t\t%d" % len(fragmentations))
    print("Number passed 2D filters: \t%d" % len(fragmentations_filt))


    # Write data to file
    # Format: full_mol (SMILES), linker (SMILES), fragments (SMILES)
    with open(args.output_path, 'w') as f:
        if args.linker_design:
            for fragmentation in fragmentations_filt:
                f.write("%s %s %s\n" % (fragmentation[0], fragmentation[1], fragmentation[2]+'.'+fragmentation[3]))
        else:
            for fragmentation in fragmentations_filt:
                f.write("%s %s %s\n" % (fragmentation[0], fragmentation[1], fragmentation[2]))

            
    print("Done!")
