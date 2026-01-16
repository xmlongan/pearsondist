import math
from ajdmom import Poly
from ajdmom.ito_mom import moment_IEII


def moment_IdIE(i, j):
    poly = moment_IEII(j, i, 0)
    # keyfor =
    # ('e^{k(n-1)h}', 'e^{k[t-(n-1)h]}', '[t-(n-1)h]', 'v_{n-1}',
    #  'k^{-}', 'theta', 'sigma_v')
    poln = Poly()
    poln.set_keyfor(['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'theta', 'sigma_v'])
    for k in poly:
        if k[0] != j and poly[k] != 0:
            raise ValueError(f"k[0] = {k[0]}, j = {j}, they are not equal! i = {i}")
        key = (k[0] - k[1],) + k[2:]
        val = poly[k]
        poln.add_keyval(key, val)
    return poln


def poly2num(poly, par):
    # ['e^{-kh}', 'h', 'v_{n-1}', 'k^{-}', 'theta', 'sigma_v']
    v_0, k, h, theta, sigma_v = par['v0'], par['k'], par['h'], par['theta'], par['sigma']
    value = 0
    for K in poly:
        val = poly[K] * math.exp(-K[0] * k * h) * (h ** K[1]) * (v_0 ** K[2])
        val *= (k ** (-K[3])) * (theta ** K[4]) * (sigma_v ** K[5])
        value += val
    return value


def con_m(i, j, par):
    mom = moment_IdIE(i, j)
    # mom = moment_IIE(i, j)
    value = poly2num(mom, par)
    return value


def comp_joint_moms_mat(par):
    mu = [[con_m(i, j, par) for j in range(10 - i + 1)] for i in range(10 + 1)]
    return mu


if __name__ == '__main__':
    par = {'h': 1, 'v0': 0.007569, 'k': 3.46, 'theta': 0.008, 'sigma': 0.14, 'rho': -0.82, 'mu': 0.0789}
    jmom = comp_joint_moms_mat(par)
