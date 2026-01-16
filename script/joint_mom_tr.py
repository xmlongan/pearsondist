import math

def joint_mom_tr(n, m, jmoms):
    """Joint moments of the transformed 2D variables

    :param int n: order of zeta1
    :param int m: order of zeta2
    :param jmoms: joint moments of the original 2D variables
    :return: joint moment value of the transformed 2D variables
    :rtype: float
    """
    m1 = jmoms[1][0]
    m2 = jmoms[0][1]
    cov = jmoms[1][1] - m1 * m2
    var = jmoms[2][0] - m1 ** 2
    c = - cov / var
    f = 0
    for i in range(m + 1):
        j = m - i
        bino = math.comb(m, i)
        f += bino * c ** i * jmoms[n + i][j]
    return f


if __name__ == '__main__':
    from joint_mom import comp_joint_moms_mat

    jmom_tr = joint_mom_tr
    degree = 4
    par = {'h': 1, 'v0': 0.007569, 'k': 3.46, 'theta': 0.008, 'sigma': 0.14, 'rho': -0.82, 'mu': 0.0789}
    jmom = comp_joint_moms_mat(par)

    mu_d1 = [jmom_tr(i, 0, jmom) for i in range(1, 2 * degree + 1)]
    mu_d2 = [jmom_tr(0, i, jmom) for i in range(1, 2 * degree + 1)]
    mu_d1d2 = [[jmom_tr(i, j, jmom) for j in range(degree + 1)] for i in range(degree + 1)]
