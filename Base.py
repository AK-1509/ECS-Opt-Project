from rdkit import Chem 
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem.rdchem import RWMol
import numpy as np

def calculate_logP(mol):
    """Calculate logP of the given molecule."""
    return Crippen.MolLogP(mol)

def calculate_molecular_size(mol):
    """Calculate the molecular size (volume and diameter) of the given molecule."""
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    mol_volume = rdMolDescriptors.CalcExactMolWt(mol)
    conf = mol.GetConformer()
    atom_positions = conf.GetPositions()
    distances = np.linalg.norm(atom_positions[:, np.newaxis] - atom_positions, axis=-1)
    max_diameter = np.max(distances)
    
    return mol_volume, max_diameter

def attach_moiety(base_mol, moiety):
    """Attach a moiety to the base molecule."""
    base_rw_mol = RWMol(base_mol)
    combined_mol = Chem.CombineMols(base_rw_mol, moiety)
    rw_combined_mol = RWMol(combined_mol)

    # Attach at the first atom of base molecule and first atom of moiety
    base_atom_idx = 0
    moiety_atom_idx = base_rw_mol.GetNumAtoms()
    
    rw_combined_mol.AddBond(base_atom_idx, moiety_atom_idx, Chem.BondType.SINGLE)
    Chem.SanitizeMol(rw_combined_mol)
    return rw_combined_mol.GetMol()

def generate_molecules(base_smiles, moiety_smiles_list):
    """Generate molecules by attaching different moieties to the base molecule."""
    base_mol = Chem.MolFromSmiles(base_smiles)
    all_molecules = []

    for moiety_smiles in moiety_smiles_list:
        moiety = Chem.MolFromSmiles(moiety_smiles)

        # Generate combined molecule
        combined_mol = attach_moiety(base_mol, moiety)
        
        # Calculate properties
        mol_wt = rdMolDescriptors.CalcExactMolWt(combined_mol)
        logP = calculate_logP(combined_mol)
        volume, diameter = calculate_molecular_size(combined_mol)

        # Append the properties to the list
        all_molecules.append((combined_mol, mol_wt, logP, volume, diameter))
    
    return all_molecules

# Define base molecule and moieties
base_smiles = "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O"
moiety_smiles_list = ["N", "C", "CC", "CCC", "O", "COC", "CCOC", "C=O", "C(=O)O", "CN", "CCN", "C#N", "C(=O)N", "CC(C)C", "N(C)C", "CCCCN", "C1CCCCC1", "c1ccccc1N", "CCO", "CCCC", "CCC", "CC(C)C", "N(CC)C", "COC", "CCOCC"]



# Generate all molecules
all_molecules = generate_molecules(base_smiles, moiety_smiles_list)

# Sort molecules by logP and select the top 5
sorted_molecules = sorted(all_molecules, key=lambda x: x[2])  # Sort by logP (index 2)
top_5_molecules = sorted_molecules[:5]

# Display the top 5 molecules with the lowest logP
for mol, mol_wt, logP, volume, diameter in top_5_molecules:
    print(f"SMILES: {Chem.MolToSmiles(mol)}, Molecular Weight: {mol_wt:.2f}, logP: {logP:.2f}, Volume: {volume:.2f} Å³, Diameter: {diameter:.2f} Å")

