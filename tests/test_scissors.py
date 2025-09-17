# tests/test_scissors.py
import pytest
from ase.build import molecule
from fflame.cell.scissors import LigandScissor

def test_ligand_scissor_init():
    water = molecule("H2O")
    scissor = LigandScissor(water)
    assert scissor.num_linkers() >= 0  # Just a sanity check

def test_extract_linker_returns_structure():
    water = molecule("H2O")
    scissor = LigandScissor(water)
    if scissor.num_linkers() > 0:
        lk = scissor.extract_linkers(0, threshold=2.0, returntype="pym")
        assert lk.__class__.__name__ == "Structure"
