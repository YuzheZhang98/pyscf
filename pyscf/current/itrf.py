'''
Helper functions that modify the QM methods to constrain the current.
'''

import numpy
import scipy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.lib import logger

def current_constraint_for_scf(scf_method,
                               current_direction=numpy.array([0., 0., 1.]),
                               current_amplitude=0.0,
                               length=10.):
    assert (isinstance(scf_method, (scf.hf.SCF, mcscf.casci.CASCI)))

    if isinstance(scf_method, scf.hf.SCF):
        # Avoid to initialize _CurrentConstraint twice
        if isinstance(scf_method, _CurrentConstraint):
            scf_method.current_direction = \
                    numpy.asarray(current_direction, dtype=numpy.double)
            scf_method.current_direction /= \
                    numpy.linalg.norm(scf_method.current_direction)
            scf_method.current_amplitude = current_amplitude
            scf_method.length = length
            return scf_method

        method_class = scf_method.__class__

    else:
        if isinstance(scf_method._scf, _CurrentConstraint):
            scf_method._scf.current_direction = \
                    numpy.asarray(current_direction, dtype=numpy.double)
            scf_method._scf.current_direction /= \
                    numpy.linalg.norm(scf_method._scf.current_direction)
            scf_method._scf.current_amplitude = current_amplitude
            scf_method._scf.length = length
            return scf_method

        method_class = scf_method._scf.__class__

    class CurrentConstraint(_CurrentConstraint, method_class):
        def __init__(self, scf_method, current_direction=numpy.array([0., 0., 1.]),
                     current_amplitude=0.0, length=10.):
            self.__dict__.update(scf_method.__dict__)
            self.current_direction = \
                    numpy.asarray(current_direction, dtype=numpy.double)
            self.current_direction /= numpy.linalg.norm(self.current_direction)
            self.current_amplitude = current_amplitude
            self.length = length
            self.current_lagrange = 0.
            # direction \cdot <i|nabla j> / length
            ipovlp = self.mol.intor('int1e_ipovlp').transpose(0, 2, 1)
            self.current_ao = 1 / length \
                    * numpy.einsum('xij,x->ij', ipovlp, self.current_direction)
            self._keys.update(['current_direction', 'current_amplitude',
                               'length', 'current_lagrange', 'current_ao'])

        def dump_flags(self, verbose=None):
            method_class.dump_flags(self, verbose)
            logger.info(self, '** Add current constraint for %s **',
                        method_class)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Direction: %.9g   %.9g  %.9g',
                             self.current_direction[0],
                             self.current_direction[1],
                             self.current_direction[2])
                logger.debug(self, 'Amplitude: %.9g',
                             self.current_amplitude)
            return self

        def get_fock(self, *args, **kwargs):
            return method_class.get_fock(self, *args, **kwargs) + \
                   0.5j * self.current_lagrange * self.current_ao

        def current_analysis(self):
            dm = self.make_rdm1()
            if dm.ndim > 2: # UHF/UKS
                total_dm = dm[0] + dm[1]
            elif dm.shape[0] == 2 * self.current_ao.shape[0]: # GHF
                nao = self.current_ao.shape[0]
                total_dm = numpy.zeros((nao, nao))
                total_dm = dm[:nao,:nao]
                total_dm += dm[nao:,nao:]
            elif dm.shape[0] != self.current_ao.shape[0]:
                raise RuntimeError(f'DM shape {dm.shape}, AO matrix shape {self.current_ao.shape}')
            else:
                total_dm = dm
            current_diff = numpy.einsum('ij,ji->', self.current_ao,
                                        total_dm.imag) - self.current_amplitude
            logger.debug(self, 'Current deviation: %.9g', current_diff)
            return current_diff

        def scf(self, *args, **kwargs):
            def f(c):
                self.current_lagrange = c
                method_class.scf(self, *args, **kwargs)
                return self.current_analysis()
            scipy.optimize.root(f, 0.0, method='hybr')
            return self.e_tot
        kernel = lib.alias(scf, alias_name='kernel')

    if isinstance(scf_method, scf.hf.SCF):
        return CurrentConstraint(scf_method, current_direction,
                                 current_amplitude, length)
    else:  # post-HF methods
        scf_method._scf = CurrentConstraint(scf_method._scf, current_direction,
                                            current_amplitude, length).run()
        scf_method.mo_coeff = scf_method._scf.mo_coeff
        scf_method.mo_energy = scf_method._scf.mo_energy
        return scf_method

# A tag to label the derived class
class _CurrentConstraint:
    pass
