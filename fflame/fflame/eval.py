from mace.calculators import MACECalculator
from ase.io import read, write
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import matplotlib as mpl
from glob import glob
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from collections import OrderedDict

E0_fflame = {
    'H': -12.6261565,
    'C': -146.3743668,
    'O': -431.60304518,
    'Al': -52.8830503,
    'N': -265.928468,
    'Mg': -1720.7269521918522,
    'Fe':-3357.3573350016,
    'Co':-3952.8376259419997,
    'Ni':-4602.329443782801,
    'Cu': -1306.1311835242948,
    'Zn': -1645.9184288744775
}

E0_mace = {
    'H': -1.11734008,
    'C': -1.26173507,
    'O': -1.54838784,
    'Al': -0.21630193,
    'Cu': -0.60025846,
    'Zn': -0.1651332,
}

def model_eval(trajfile, outputfile, macemodellist, device):
    atoms = read(trajfile,':')
    for idx, model in enumerate(macemodellist):
        if 'cuda' in device:
            mace_calc = MACECalculator(model_paths=[model], device=device, enable_cueq=True)
        else:
            mace_calc = MACECalculator(model_paths=[model], device=device)
        for atom in tqdm(atoms):
            atom.calc = mace_calc
            atom.info[f'energy_mace{idx}'] = atom.get_potential_energy()
            atom.arrays[f'force_mace{idx}'] = atom.calc.get_forces()
            atom.info[f'stress_mace{idx}'] = atom.get_stress()
            atom.calc = None
    write(outputfile, atoms)

def baseline_shift(atoms, E0_fflame, E0_mace):
    formula = Counter(atoms.get_chemical_symbols())
    shift = 0.0
    for elem, count in formula.items():
        if elem in E0_fflame and elem in E0_mace:
            shift += count * (E0_fflame[elem] - E0_mace[elem])
        else:
            raise ValueError(f"Missing baseline for element {elem}")
    return shift