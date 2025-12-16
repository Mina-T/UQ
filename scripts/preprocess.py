# Converts .xyz / .extxyz files into LATTE-compatible .example JSON files

import os
import json

import numpy as np
import torch

from ase import io
from ase.io import extxyz

ptable = ['null', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']

def get_energy(Atoms):
    for key in ['REF_energy', 'Energy', 'energy[eV]']:
        if key in Atoms.info:
            return Atoms.info[key]
    try:
        return Atoms.get_potential_energy()
    except Exception:
        return 1000.0


def get_forces(Atoms):
    for key in ['forceseV/Ang', 'REF_forces', 'force']:
        if key in Atoms.arrays:
            return np.asarray(Atoms.arrays[key])
    try:
        return np.asarray(Atoms.get_forces())
    except Exception:
        return np.zeros((len(Atoms), 3))

def High_force_sys(forces: np.ndarray, thr: float = 11.0) -> bool:
    '''
    returns True if a system contains forces exceeding a threshold
    '''
    return np.max(forces) > thr

class xyz_to_example:
    """
    Convert extended XYZ datasets into LATTE JSON (.example) format.

    Each atomic configuration is parsed, optionally filtered by force,
    serialized to JSON, optionally reference atomic energies is estimated.
    """
    
    def __init__(self, _path: str, _file_name: str, filter_high_force: bool = False):
        self.path = _path if _path.endswith("/") else f"{_path}/"
        self.file_name = _file_name
        self.pos_unit = 'cartesian'
        self.length_unit= 'angstrom'
        self.filter_high_force = filter_high_force

    def count_atomic_structures(self) -> int:
        """Returns the number of atomic configurations in the XYZ file."""
        return len(io.read(os.path.join(self.path, self.file_name), index=":"))
    
    def read_system(self, idx):
        """
        Reads a single structure from an extended XYZ file.
        Returns:
        [positions, cell, atomic_numbers, energy, forces, category]
            or None if filtered.
        """
        structure = io.read(
            os.path.join(self.path, self.file_name),
            index=idx,
            format="extxyz")

        atomic_numbers = structure.get_atomic_numbers()
        cell = np.array(structure.get_cell())
        pos = np.array(structure.get_positions())
        energy = get_energy(structure)
        forces = get_forces(structure)
        category = structure.info.get('category', 'unknown')
        
        if self.filter_high_force and High_force_sys(forces):
            return None

        return  [pos, cell, atomic_numbers, energy, forces, category]

    

    def build_atoms(self, data: list):
        """
        Build the .example atom block
        """
        pos, _, atomic_numbers, _, forces, _ = data
        pos = pos.tolist()
        atomic_numbers = atomic_numbers.tolist()
        forces = forces.tolist()

        atoms = []
        system_atomic_symbols = [ptable[int(atom)] for atom in atomic_numbers]
        for idx, (atom, pos, force) in enumerate(zip(system_atomic_symbols, pos, forces)):
                atoms.append([idx, atom, pos, force])
    
        return atoms

    def dump_json(self, idx: int, data: list):
        """
        Write a single structure to a .example JSON file.
        """
        _, cell, _, energy, _, category= data
        json_dict = {
                "key" : idx,
                "atomic_position_unit" : self.pos_unit,
                "unit_of_length" : self.length_unit,
                "energy" : [energy, "eV"],
                "lattice_vectors" : cell.tolist(),
                "atoms" : self.build_atoms(data)
        }
        
        modified_category = category.replace("/", "_").replace(" ", "")
        fname = f"{modified_category}_{idx}.example"

        with open(fname, "w") as f:
            json.dump(json_dict, f)

        if idx % 100 == 0:
            print(f'writing {idx}th example file', flush = True)
        
    def run(self, start_idx: int = 0):
        total_structures = self.count_atomic_structures()
        output_dir = f"{self.file_name.split('.')[0]}_examples"
        output_path = os.path.join(self.path, output_dir)
        os.makedirs(output_path, exist_ok=True)
        os.chdir(output_path) 

        for idx in range(start_idx, total_structures):
            data = self.read_system(idx)
            if data is None:
                continue
            self.dump_json(idx, data)
        os.chdir(self.path)