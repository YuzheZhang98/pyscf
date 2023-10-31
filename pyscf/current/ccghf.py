#!/usr/bin/env python

'''
Current-Constrained Generalized Hartree-Fock (CC-GHF)
'''

import numpy
from pyscf.scf.ghf import GHF


def get_fock_add_ccghf(ipovlp, c, length, direct):
    f_add = (.5j * c / length) * ipovlp
    return numpy.einsum('xij,x->ij', f_add, direct)


class CCGHF(GHF):
    def __init__(self, mol, c=0., current_dir=[0., 0., 1.], length=10):
        GHF.__init__(self, mol)
        self.c = c
        self.length = length
        current_dir = numpy.asarray(current_dir, dtype=numpy.double)
        current_dir /= numpy.linalg.norm(current_dir)
        self.current_dir = current_dir
        # <i|nabla j>
        block = mol.intor('int1e_ipovlp').transpose(0, 2, 1)
        self.ipovlp = numpy.block([[block, numpy.zeros(block.shape)],
                                   [numpy.zeros(block.shape), block]])
        self._keys.update(['c', 'length', 'current_dir', 'ipovlp'])

    def get_fock_add_ccghf(self):
        return get_fock_add_ccghf(self.ipovlp, self.c,
                                  self.length, self.current_dir)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None):
        f0 = GHF.get_fock(self, h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=cycle,
                          diis=diis, diis_start_cycle=diis_start_cycle,
                          level_shift_factor=level_shift_factor,
                          damp_factor=damp_factor)  # Fock without constraint
        return f0 + self.get_fock_add_ccghf()
