import unittest
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import mcscf
from pyscf.current import itrf

def setUpModule():
    global mol
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = ''' H    0.000   0.000    0.
                   H    0.000   0.000    0.74''',
        basis = 'cc-pvdz')

def tearDownModule():
    global mol

class KnowValues(unittest.TestCase):
    # no current
    def test_energy_no_current(self):
        mf = itrf.current_constraint_for_scf(scf.RHF(mol),
                                             current_direction=numpy.array([0., 0., 1.]),
                                             current_amplitude=0.,
                                             length=(0.74+1.2*2) * 1.8897259886)
        self.assertAlmostEqual(mf.kernel(), scf.RHF(mol).kernel(), 9)
    # 1e-3 current
    def test_energy_small_current(self):
        mf = itrf.current_constraint_for_scf(scf.RHF(mol),
                                             current_direction=numpy.array([0., 0., 1.]),
                                             current_amplitude=1e-3,
                                             length=(0.74+1.2*2) * 1.8897259886)
        self.assertAlmostEqual(mf.kernel(), -1.1286913024658398, 9)
    def test_energy_small_current_UKS(self):
        mf = itrf.current_constraint_for_scf(dft.UKS(mol),
                                             current_direction=numpy.array([0., 0., 1.]),
                                             current_amplitude=1e-3,
                                             length=(0.74+1.2*2) * 1.8897259886)
        self.assertAlmostEqual(mf.kernel(), -1.1314024771833786, 9)

if __name__ == "__main__":
    print("Full Tests for current constraint")
    unittest.main()
