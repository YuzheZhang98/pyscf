#!/usr/bin/env python

import math
from pyscf import gto
from pyscf.lib import logger

class Mole(gto.mole.Mole):
    '''A subclass of gto.mole.Mole to handle quantum nuclei in NEO.
    By default, all atoms would be treated quantum mechanically.

    Example:

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; C 0 0 1.1; N 0 0 2.2', quantum_nuc = [0,1], basis = 'ccpvdz')
    # H and C would be treated quantum mechanically
    >>> mol.build(atom = 'H 0 0 0; C 0 0 1.1; N 0 0 2.2', basis = 'ccpvdz')
    # All atoms are treated quantum mechanically by default

    '''

    def __init__(self, **kwargs):
        gto.mole.Mole.__init__(self, **kwargs)

        self.quantum_nuc = [] # a list to assign which nuclei are treated quantum mechanically
        self.nuc_num = 0 # the number of quantum nuclei
        self.mass = [] # the mass of nuclei
        self.elec = None # a Mole object for NEO-electron and classical nuclei
        self.nuc = [] # a list of Mole objects for quantum nuclei

    def elec_mole(self):
        'return a Mole object for NEO-electron and classical nuclei'

        elec = gto.mole.copy(self)
        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= elec._atm[i,0]
                elec._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0
        elec.charge += quantum_nuclear_charge # charge determines the number of electrons
        return elec

    def nuc_mole(self, atom_index):
        '''
        Return a Mole object for specified quantum nuclei. Default basis is even-tempered Gaussian basis.

        H: 8s8p8d alpha=2*sqrt(2) beta=sqrt(2)
        other heavy atom: 12s12p12d alpha=2*sqrt(2)*mass beta=sqrt(3)
        '''

        nuc = gto.mole.copy(self) # a Mole object for quantum nuclei
        nuc.atom_index = atom_index

        # even-tempered basis
        alpha = 2*math.sqrt(2)*self.mass[atom_index]

        if self.atom_symbol(atom_index) == 'H@2':
            basis = gto.basis.parse('''
                    H   S
                    2.767   1.000
                    H   S
                    12.350   1.000
                    H   S
                    22.638  1.000
                    H   S
                    45.244  1.000
                    H   P
                    13.345   1.000
                    H   P
                    19.506  1.000
                    H   P
                    33.976  1.000
                    H   D
                    14.881   1.000
                    H   D
                    26.889  1.000
                    ''')
        elif self.atom_pure_symbol(atom_index) == 'H':
            basis = gto.basis.parse('''
                    H   S
                    1.957   1.000
                    H   S
                    8.734   1.000
                    H   S
                    16.010  1.000
                    H   S
                    31.997  1.000
                    H   P
                    9.438   1.000
                    H   P
                    13.795  1.000
                    H   P
                    24.028  1.000
                    H   D
                    10.524   1.000
                    H   D
                    19.016  1.000
                    ''')
        else:
            beta = math.sqrt(3)
            n = 12
            basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta)])
            #logger.info(self, 'Nuclear basis for %s: n %s alpha %s beta %s' %(self.atom_symbol(atom_index), n, alpha, beta))

        nuc._basis = gto.mole.format_basis({self.atom_symbol(atom_index): basis})
        nuc._atm, nuc._bas, nuc._env = gto.mole.make_env(nuc._atom, nuc._basis, self._env[:gto.PTR_ENV_START])
        quantum_nuclear_charge = 0
        for i in range(len(self.quantum_nuc)):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= nuc._atm[i,0]
                nuc._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0

        nuc.charge += quantum_nuclear_charge
        nuc.spin = 0
        nuc.nelectron = 2 # avoid UHF

        return nuc

    def build(self, quantum_nuc = 'all', nuc_basis = 'etbs', **kwargs):
        'assign which nuclei are treated quantum mechanically by quantum_nuc (list)'
        gto.mole.Mole.build(self, **kwargs)

        self.quantum_nuc = [False]*self.natm

        if quantum_nuc is 'all':
            self.quantum_nuc = [True]*self.natm
            logger.info(self, 'All atoms are treated quantum-mechanically by default.')
        elif isinstance(quantum_nuc, list):
            for i in quantum_nuc:
                self.quantum_nuc[i] = True
                logger.info(self, 'The %s(%i) atom is treated quantum-mechanically' %(self.atom_symbol(i), i))
        else:
            raise TypeError('Unsupported parameter %s' %(quantum_nuc))

        self.nuc_num = len([i for i in self.quantum_nuc if i == True])

        self.mass = self.atom_mass_list(isotope_avg=True)
        for i in range(len(self.mass)):
            if self.atom_symbol(i) == 'H@2': # Deuterium (from Wikipedia)
                self.mass[i] = 2.01410177811
            elif self.atom_symbol(i) == 'H@0': # Muonium (TODO: precise mass)
                self.mass[i] = 0.114
            elif self.atom_pure_symbol(i) == 'H': # Proton (from Wikipedia)
                self.mass[i] = 1.007276466621

        self.elec = self.elec_mole()
        self.nuc = []
        for i in range(len(self.quantum_nuc)):
            if self.quantum_nuc[i] == True:
                self.nuc.append(self.nuc_mole(i))
