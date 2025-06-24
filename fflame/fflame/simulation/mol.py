from xtb.ase.calculator import XTB
from ase.optimize.lbfgs import LBFGS
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import os
import time
import numpy as np
import pylab as pl
from ase.io import read, write
from ase.md import MDLogger

class XYZTrajectoryWriter:
    def __init__(self, atoms, filename="trajectory.xyz", interval=5):
        self.atoms = atoms
        self.filename = filename
        self.interval = interval
        self.counter = 0
        # Clear file at the start
        open(self.filename, 'w').close()

    def __call__(self):
        self.counter += 1
        if self.counter % self.interval == 0:
            write(self.filename, self.atoms, append=True)
            
def GeometryOptimization(atoms, outfile, logfile, interval=5, Fmax=0.01, stepMax=200):
    """Run geometry optimization using XTB calculator.

    Args:
        atoms (ase.Atoms): Atoms object for optimization.
        outfile (str): Output filename for the trajectory.
        interval (int): Interval for writing frames.
        Fmax (float): Maximum force for convergence.
        stepMax (int): Maximum number of optimization steps.
    """

    calc = XTB(method = 'GFN1-xTB')
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=logfile)
    opt.attach(XYZTrajectoryWriter(atoms, filename=outfile, interval=interval))
    
    return opt.run(fmax=Fmax, steps=stepMax)