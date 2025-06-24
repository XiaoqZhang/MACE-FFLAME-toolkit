from math import cos, sin, sqrt, fabs, ceil, pi
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.local_env import CutOffDictNN, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
from ase.io import read
from ase import Atoms
import yaml
import os
file_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(file_folder, "../data/tuned_vesta.yml"), "r", encoding="utf8") as handle:
    _VESTA_CUTOFFS = yaml.load(handle, Loader=yaml.UnsafeLoader)  

class LigandBuilder:
    def __init__(self, ligand, **kwargs):
        """
        Args:
        structure:
            - CIF filepath (str or Path)
            - ASE Atoms object
        """
        if isinstance(ligand, (str, Path)):
            self.atoms = read(ligand)
            if ligand.endswith(".cif"):
                self.source = "file"
            elif ligand.endswith(".xyz"):
                self.source = "xyz"
            self.filename = ligand
            self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        elif isinstance(ligand, Atoms):
            self.atoms = ligand
            self.source = "atoms"
            self.filename = None
            self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        else:
            raise TypeError("Unsupported type for ligand. Must be str, Path, or Atoms.")
        
        self.bonded_mol = self.parse_mol_cluster(**kwargs)
        
        
    def parse_mol_cluster(self, method='vesta', **kwargs):
        """
        Parse the ligand structure and return a pymatgen Structure object.
        """
        method = method.lower()
        
        if method == 'vesta':
            scaling = kwargs.get('scaling', 1)
            cutoffs = {k: v * scaling for k, v in _VESTA_CUTOFFS.items()}
            self.method = CutOffDictNN(cutoffs)
        elif method == 'crystal':
            self.method = CrystalNN(**kwargs)
        
        # Convert ASE Atoms to pymatgen Structure
        self.structure = AseAtomsAdaptor.get_structure(self.atoms)
        
        # Get the first molecule from the structure
        sg = StructureGraph.with_local_env_strategy(self.structure, self.method)
        clusters = sg.get_subgraphs_as_molecules()
        molecule = clusters[0]
        assert len(molecule) == len(self.structure)
        
        return molecule

    def radius(self, mol):
        """
        Calculate the radius of the molecule.
        """
        coords = np.array([site.coords for site in mol.sites])
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        return distances.max()
    
    def put_mol_in_lattice(self, lattice_type, log=True, **kwargs):
        """Put the molecule in the lattice.

        Args:
            lattice_type (str): Lattice type, e.g., 'cubic', 'supercell'. 
            molecule (pymatgen.Molecule): The molecule to be placed in the lattice.
        """
        
        if lattice_type == 'cubic':
            cubic_method = kwargs.get('cubic_method', 'fixed')
            if cubic_method == 'fixed':
                cell = kwargs.get('cell_length', 30)
                self.lattice = Lattice.cubic(cell)
            elif cubic_method == 'auto':
                threshold = kwargs.get('threshold', 4)
                box = (self.radius(self.bonded_mol) + threshold) * 2
                self.lattice = Lattice.cubic(box)
                if log:
                    print(f'Bonded mol radius: {self.radius(self.bonded_mol)}; box size: {box}')
            if log:
                print(f"Created cubic cell with lengths {self.lattice.abc} and angles {self.lattice.angles}")
        elif lattice_type == 'supercell':
            cutoff = kwargs.get('cutoff', 10)
            cell_params = self.atoms.get_cell_lengths_and_angles()
            na, nb, nc= self.count_supercell(cell_params, cutoff)
            large_cell = np.dot(np.diag([na, nb, nc]), self.atoms.cell)
            self.lattice = Lattice(large_cell)
            if log:
                print(f"Supercell size: {na} x {nb} x {nc}; new cell lengths {self.lattice.abc} with angles {self.lattice.angles}")
        else:
            raise ValueError("Unsupported lattice type. Must be 'cubic' or 'supercell'.")
        
        # Center the molecule in the lattice
        coords = np.array([site.coords for site in self.bonded_mol.sites])
        centered_coords = coords - coords.mean(axis=0) + self.lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        
        species = [site.specie for site in self.bonded_mol.sites]
        new_structure = Structure(
            self.lattice,
            species,
            centered_coords,
            coords_are_cartesian=True
        )
        mol = AseAtomsAdaptor.get_atoms(new_structure)
        
        return mol

    def count_supercell(self, cell_params, threshold):
        """
        Count the supercell size based on the cutoff distance.

        Args:
            cell_params (dict): Cell parameters.
            cutoff (float): Cutoff distance.

        Returns:
            tuple: Supercell size (na, nb, nc).
        """
        deg2rad = pi / 180.
        a_len, b_len, c_len = cell_params[0], cell_params[1], cell_params[2]
        alpha, beta, gamma = cell_params[3] * deg2rad, cell_params[4] * deg2rad, cell_params[5] * deg2rad

        # Computing triangular cell matrix
        vol = np.sqrt(1 - cos(alpha)**2 - cos(beta)**2 - cos(gamma)**2 + 2 * cos(alpha) * cos(beta) * cos(gamma))
        cell = np.zeros((3, 3))
        cell[0, :] = [a_len, 0, 0]
        cell[1, :] = [b_len * cos(gamma), b_len * sin(gamma), 0]
        cell[2, :] = [
            c_len * cos(beta), c_len * (cos(alpha) - cos(beta) * cos(gamma)) / (sin(gamma)), c_len * vol / sin(gamma)
        ]
        cell = np.array(cell)

        # Computing perpendicular widths, as implemented in Raspa
        # for the check (simplified for triangular cell matrix)
        axc1 = cell[0, 0] * cell[2, 2]
        axc2 = -cell[0, 0] * cell[2, 1]
        bxc1 = cell[1, 1] * cell[2, 2]
        bxc2 = -cell[1, 0] * cell[2, 2]
        bxc3 = cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0]
        det = fabs(cell[0, 0] * cell[1, 1] * cell[2, 2])
        perpwidth = np.zeros(3)
        perpwidth[0] = det / sqrt(bxc1**2 + bxc2**2 + bxc3**2)
        perpwidth[1] = det / sqrt(axc1**2 + axc2**2)
        perpwidth[2] = cell[2, 2]

        thr = max(0.001, threshold)

        return int(ceil(thr / perpwidth[0])), int(ceil(thr / perpwidth[1])), int(ceil(thr / perpwidth[2]))
        
        
        

