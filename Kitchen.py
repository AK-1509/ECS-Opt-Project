# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 03:08:27 2024

@author: kalek
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import RWMol

def find_stable_attachment_points(mol):
    """
    Identify potential attachment points in the molecule based on atom properties.
    This function avoids attachment points on strained rings and highly substituted carbons.
    """
    attachment_points = []
    
    for atom in mol.GetAtoms():
        # Avoid atoms in strained rings and highly substituted sites
        if atom.GetDegree() < 3 and atom.GetHybridization() != Chem.HybridizationType.SP:
            # Only include carbons, oxygens, and nitrogens as potential attachment points
            if atom.GetSymbol() in ['C', 'O', 'N']:
                attachment_points.append(atom.GetIdx())
    
    return attachment_points

def attach_fragment_at_stable_point(base_mol, fragment, stable_points):
    """
    Attempt to attach a fragment to the base molecule at a given list of stable points.
    """
    best_mol = None
    best_score = float('inf')
    
    for point in stable_points:
        # Create a new combined molecule
        combined_mol = Chem.CombineMols(base_mol, fragment)
        rw_combined_mol = RWMol(combined_mol)
        
        # Attempt to attach the fragment at the stable point
        try:
            # Add a bond between the stable attachment point and the first atom of the fragment
            fragment_atom_idx = base_mol.GetNumAtoms()
            rw_combined_mol.AddBond(point, fragment_atom_idx, Chem.BondType.SINGLE)
            
            # Sanitize the molecule and calculate a stability score
            Chem.SanitizeMol(rw_combined_mol)
            temp_mol = rw_combined_mol.GetMol()
            
            # Calculate an empirical score (e.g., using logP)
            score = abs(Descriptors.MolLogP(temp_mol)) + Descriptors.MolWt(temp_mol)
            
            # Keep the molecule with the lowest score
            if score < best_score:
                best_mol = temp_mol
                best_score = score
        
        except Exception as e:
            # Skip if adding the bond failed
            continue
    
    return best_mol

def decompose_and_reconstruct(base_smiles, fragment_smiles):
    """
    Decomposes the base molecule, then reattaches fragments at calculated stable points.
    """
    base_mol = Chem.MolFromSmiles(base_smiles)
    fragment_mol = Chem.MolFromSmiles(fragment_smiles)
    
    if base_mol is None or fragment_mol is None:
        raise ValueError("Invalid SMILES input for base molecule or fragment.")
    
    # Find stable points in the base molecule
    stable_points = find_stable_attachment_points(base_mol)
    
    # Attach the fragment at the most stable point
    new_mol = attach_fragment_at_stable_point(base_mol, fragment_mol, stable_points)
    
    return new_mol

# Example usage
base_smiles = "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O"
fragment_smiles = "NC(=O)CC"  # Example fragment to add

# Decompose and reconstruct the molecule
new_molecule = decompose_and_reconstruct(base_smiles, fragment_smiles)

if new_molecule:
    print("Reconstructed Molecule SMILES:", Chem.MolToSmiles(new_molecule))
else:
    print("Failed to find a stable attachment point for the fragment.")
