"""
Example
-------
>>> from ase.io import read
>>> from fflame.cell.scissors import LigandScissor
>>> 
>>> mof = read("HKUST1.cif")
>>> scissors = LigandScissor(mof)
>>> linker = scissors.extract_linkers(index=0, threshold=2.0, returntype="ase")
>>> flag, optimized_linker = scissors.optimize_linker(linker, keep_origin=False, method="vesta")
"""

from moffragmentor import MOF
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Lattice
from ase import Atoms
import numpy as np
import time
from mofchecker.checks.local_structure.add_hydrogen import O_site_adding_hydrogen
from xtb.ase.calculator import XTB
from ase.optimize.lbfgs import LBFGS
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import yaml
from pymatgen.analysis.local_env import CutOffDictNN, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
import os
import random
file_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(file_folder, "../data/tuned_vesta.yml"), "r", encoding="utf8") as handle:
    _VESTA_CUTOFFS = yaml.load(handle, Loader=yaml.UnsafeLoader)  

class LigandScissor:
    """
    A class for extracting and optimizing linker fragments from MOFs.

    Parameters
    ----------
    mof : pymatgen.Structure or ase.Atoms
        A MOF structure from which linkers will be extracted.

    Raises
    ------
    TypeError
        If `mof` is not a supported type.
    """
    def __init__(self, mof: Union[Structure, Atoms]):
        if isinstance(mof, Structure) or isinstance(mof, Atoms):
            random_name = os.path.join(file_folder, f"{int(time.time())}-{random.randint(0, 2**32 - 1)}.cif")
            write(random_name, mof)
            self.mof = MOF.from_cif(random_name)
            os.remove(random_name)
        else:
            raise TypeError("Unsupported type for mof. ")
        
    def fragments(self):
        """Return the MOF fragments produced by `moffragmentor`."""
        return self.mof.fragment()
    
    def num_linkers(self):
        """Return the number of linkers in the MOF."""
        return len(self.fragments().linkers)
    
    def linkers(self):
        """Return the list of linkers extracted by `moffragmentor`."""
        return self.fragments().linkers   
    
    def lattice(self):
        """Return the lattice of the MOF structure."""
        return self.mof.structure.lattice
    
    def extract_linkers(
        self, index: int, threshold: float, returntype: str
    ) -> Union[Structure, Atoms]:
        """
        Extract a linker molecule from the MOF.

        Parameters
        ----------
        index : int
            Index of the linker to extract.
        threshold : float
            Padding distance (Å) around the extracted linker.
        returntype : {"pym", "ase"}
            Output type: "pym" for pymatgen Structure, "ase" for ASE Atoms.

        Returns
        -------
        Structure or Atoms
            The extracted linker in the chosen format.

        Raises
        ------
        ValueError
            If `returntype` is not one of {"pym", "ase"}.
        """
        lkmol = self.linkers()[index].molecule
        
        coords = np.array([site.coords for site in lkmol.sites])
        center = coords.mean(axis=0)
        radius = np.linalg.norm(coords - center, axis=1).max()
        box = (radius + threshold) * 2
        lklattice = Lattice.cubic(box)
        centered_coords = coords - center + lklattice.get_cartesian_coords([0.5, 0.5, 0.5])
        
        lkstruc = Structure(
            lklattice, 
            [site.species_string for site in lkmol], 
            coords=centered_coords, 
            coords_are_cartesian=True
        )
            
        if returntype == "pym":      
            return lkstruc
        elif returntype == "ase":
            lkatom = AseAtomsAdaptor.get_atoms(lkstruc)
            return lkatom
        
    def optimize_linker(
        self, extracted_linker: Union[Structure, Atoms], keep_origin: bool, method: str, **kwargs
    ) -> Union[Tuple[bool, Atoms], Tuple[bool, Atoms, Atoms]]:
        """
        Optimize an extracted linker by adding hydrogens and relaxing geometry with XTB.

        Parameters
        ----------
        extracted_linker : Structure or Atoms
            The extracted linker structure to optimize.
        keep_origin : bool
            Whether to also return the original, unmodified structure.
        method : {"vesta", "crystal"}
            Method used for neighbor graph construction:
            - "vesta" : Use tuned VESTA cutoffs.
            - "crystal" : Use CrystalNN environment.
        **kwargs :
            Extra arguments passed to the graph constructor.
            For "vesta", accepts "scaling" (float) to scale cutoff values.

        Returns
        -------
        tuple
            If keep_origin=True:
                (flag, optimized_atoms, original_atoms)
            Else:
                (flag, optimized_atoms)
        """
        
        if isinstance(extracted_linker, Atoms):
            extracted_linker = AseAtomsAdaptor.get_structure(extracted_linker)
        old_natom = len(extracted_linker)
        
        if method == 'vesta':
            scaling = kwargs.get('scaling', 1)
            cutoffs = {k: v * scaling for k, v in _VESTA_CUTOFFS.items()}
            graph_method = CutOffDictNN(cutoffs)
        elif method == 'crystal':
            graph_method = CrystalNN(**kwargs)
        else:
            raise ValueError("Invalid method. Choose 'vesta' or 'crystal'.")
            
        graph_linker = StructureGraph.with_local_env_strategy(extracted_linker, graph_method)
        _, h_pos = O_site_adding_hydrogen(extracted_linker, graph_linker)._get_O_charge()
        for coord in h_pos:
            extracted_linker.append("H", coord, coords_are_cartesian=True)
        new_natom = len(extracted_linker)
        
        print(f"{new_natom-old_natom} hydrogens are added! ")
        
        if keep_origin:
            ori_atom = AseAtomsAdaptor.get_atoms(extracted_linker)
        lkatom = AseAtomsAdaptor.get_atoms(extracted_linker)
        
        lkatom.calc = XTB(method = 'GFN1-xTB')
        constraint = FixAtoms(indices=np.arange(old_natom))
        lkatom.set_constraint(constraint)
        
        MaxwellBoltzmannDistribution(lkatom, temperature_K=300)
        dyn = Langevin(lkatom, 1.0*units.fs, temperature_K=300, friction=0.1)
        dyn.run(50)
        
        optimizer = LBFGS(lkatom)
        # optimizer.attach(XYZTrajectoryWriter(lkatom, filename='hydrogen_opt.xyz', interval=1))
        flag = optimizer.run(fmax=0.01, steps=200)

        if keep_origin:
            return flag, lkatom, ori_atom
        else:
            return flag, lkatom
