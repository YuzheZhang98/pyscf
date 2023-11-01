from pyscf import gto, current, scf
import numpy
import scipy


def current_analysis(c, amp, mol, direct, length):
    mf = current.CCUHF(mol, c=c, current_dir=direct, length=length)
    mf.kernel()
    dm = mf.make_rdm1()
    j = 1 / length * mf.ipovlp
    if dm.ndim > 2:
        total_dm = dm[0] + dm[1]
    else:
        total_dm = dm
    currentI = numpy.einsum('xij,ji->x', j, total_dm.imag)
    diff = numpy.dot(currentI, direct) - amp
    print(f'Lagrange multiplier:{c}')
    print(f'Current deviation: {diff}')
    return diff


unit_covert = 6.623618237510e-3  # Amp/au
amp = 5e-6 / unit_covert
mol = gto.M(atom='3-acene.xyz', basis='6-31g', spin=0)
length = 7.33295 / 0.529177
direct = [1., 0., 0.]

scipy.optimize.root(current_analysis, 0.0,
                    args=(amp, mol, direct, length), method='hybr')
