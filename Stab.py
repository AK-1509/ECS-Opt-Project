from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw

def calculate_drug_likeness(smiles: str) -> float:
    """
    Calculate drug-likeness based on multiple criteria.
    Returns a continuous score from 0 to 4.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0

    logP = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
    num_h_bond_donors = rdMolDescriptors.CalcNumHBD(mol)
    num_h_bond_acceptors = rdMolDescriptors.CalcNumHBA(mol)
    mol_wt = Descriptors.MolWt(mol)

    # Score calculations
    score = (1 / (1 + logP)) + \
            (1 / (1 + num_h_bond_donors)) + \
            (1 / (1 + num_h_bond_acceptors)) + \
            (1 / (1 + mol_wt / 500))

    return score

def attach_moiety(base_mol, moiety):
    """
    Attach a moiety to the base molecule and return the combined molecule.
    This function will check the valence of the atoms to prevent valence violations.
    """
    base_rw_mol = Chem.RWMol(base_mol)
    combined_mol = Chem.CombineMols(base_rw_mol, moiety)
    rw_combined_mol = Chem.RWMol(combined_mol)

    # Find a suitable attachment point by checking valences
    for atom_idx in range(base_rw_mol.GetNumAtoms()):
        atom = base_rw_mol.GetAtomWithIdx(atom_idx)
        if atom.GetFormalCharge() < 0:  # Skip negatively charged atoms
            continue

        # Check if the atom has less than the maximum allowed bonds
        if atom.GetDegree() < atom.GetImplicitValence() + atom.GetFormalCharge():
            moiety_atom_idx = base_rw_mol.GetNumAtoms()  # The first atom in the moiety

            # Add a bond between the base molecule atom and the moiety atom
            rw_combined_mol.AddBond(atom_idx, moiety_atom_idx, Chem.BondType.SINGLE)

            # Sanitize the molecule to finalize bonding
            Chem.SanitizeMol(rw_combined_mol)
            return rw_combined_mol.GetMol()

    raise ValueError("No suitable attachment point found that adheres to valency rules.")

def generate_and_filter_molecules(base_smiles, moiety_smiles_list, min_molecular_weight, max_molecular_weight):
    base_mol = Chem.MolFromSmiles(base_smiles)
    filtered_molecules = []

    # Iterate over each moiety
    for moiety_smiles in moiety_smiles_list:
        moiety = Chem.MolFromSmiles(moiety_smiles)

        # Generate new molecule by attaching the moiety
        try:
            combined_mol = attach_moiety(base_mol, moiety)
            combined_smiles = Chem.MolToSmiles(combined_mol)

            # Calculate drug-likeness and logP
            drug_likeness = calculate_drug_likeness(combined_smiles)
            mol_wt = Descriptors.MolWt(combined_mol)
            logP = rdMolDescriptors.CalcCrippenDescriptors(combined_mol)[0]

            # Apply filtering criteria for drug-likeness, molecular weight range, and logP
            if (drug_likeness > 1.5 and 
                min_molecular_weight <= mol_wt <= max_molecular_weight and 
                logP <= 5):  
                filtered_molecules.append((combined_mol, combined_smiles, drug_likeness, mol_wt, logP))
        except ValueError as e:
            print(f"Skipping moiety {moiety_smiles}: {e}")

    return filtered_molecules

# Define base molecule and expanded moieties
base_smiles = "CCC1=CCCC2=C(C1=CC(C)=C2)C(=C(C)C)C(=O)O"
moiety_smiles_list = [
    "C(C)(C)O",   # tert-Butanol
    "NC(C)C",     # Isopropylamine
    "C1CCCCC1",   # Cyclohexane
    "C(C)(C)C",   # Neopentane
    "C(=O)O",     # Carboxylic Acid
    "C(=O)N",     # Amide
    "C=CC",       # Allyl group
    "c1ccccc1",   # Phenyl group
    "C#N",        # Nitrile
    "N#C",        # Isocyanide
    "O=C(C)C",    # Acetone
    "S(=O)(=O)O", # Sulfonic Acid
    "Cl",         # Chlorine
    "Br",         # Bromine
    "F",          # Fluorine
]

# Define molecular weight range
min_molecular_weight = 300  # Minimum molecular weight
max_molecular_weight = 500  # Maximum molecular weight

# Generate and filter molecules
filtered_molecules = generate_and_filter_molecules(base_smiles, moiety_smiles_list, min_molecular_weight, max_molecular_weight)

# Display the filtered molecules with their drug-likeness scores and logP
for mol, smiles, score, mol_wt, logP in filtered_molecules:
    print(f"SMILES: {smiles}, Drug-likeness Score: {score:.2f}, Molecular Weight: {mol_wt:.2f}, logP: {logP:.2f}")
