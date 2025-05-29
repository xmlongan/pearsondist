import numpy as np


def is_conj(z1, z2, tol=1e-10):
    return (np.isclose(z1.real, z2.real, atol=tol) and
            np.isclose(z1.imag, -z2.imag, atol=tol))


def is_equal(z1, z2, tol=1e-10):
    return (np.isclose(z1.real, z2.real, atol=tol) and
            np.isclose(z1.imag, z2.imag, atol=tol))


class RootCatalog4:
    ordered_z = None
    type_no = None

    def __init__(self, z, eps=1e-10):
        indicator = np.abs(z.imag) < eps
        rroot, croot = z[indicator], z[~indicator]
        if len(croot) == 4:
            # two pairs of conjugates
            if is_conj(z[0], z[1]):
                if is_equal(z[0], z[2]) or is_equal(z[0], z[3]):
                    self.ordered_z = z
                    self.type_no = 41
                else:
                    self.ordered_z = z
                    self.type_no = 42
            elif is_conj(z[0], z[2]):
                if is_equal(z[0], z[1]) or is_equal(z[0], z[3]):
                    self.ordered_z = z[[0, 2, 1, 3]]
                    self.type_no = 41
                else:
                    self.ordered_z = z[[0, 2, 1, 3]]
                    self.type_no = 42
            else:
                if is_equal(z[0], z[1]) or is_equal(z[0], z[2]):
                    self.ordered_z = z[[0, 3, 1, 2]]
                    self.type_no = 41
                else:
                    self.ordered_z = z[[0, 3, 1, 2]]
                    self.type_no = 42
        elif len(croot) == 2:
            # one pair of conjugates
            if abs(rroot[0].real - rroot[1].real) < eps:
                self.ordered_z = np.array([rroot[0], rroot[1], croot[0], croot[1]])
                self.type_no = 43
            else:
                if rroot[0].real < rroot[1].real:
                    self.ordered_z = np.array([rroot[0], rroot[1], croot[0], croot[1]])
                    self.type_no = 44
                else:
                    self.ordered_z = np.array([rroot[1], rroot[0], croot[0], croot[1]])
                    self.type_no = 44
        else:
            # all real roots
            rt = np.sort(rroot.real)  # rt[0] <= rt[1]
            if abs(rt[0] - rt[1]) < eps:
                if abs(rt[1] - rt[2]) < eps:
                    if abs(rt[2] - rt[3]) < eps:
                        # x1 = x2 = x3 = x4
                        self.ordered_z = rt
                        self.type_no = 45
                    else:
                        # x1 = x2 = x3 != x4
                        self.ordered_z = rt
                        self.type_no = 46
                else:
                    if abs(rt[2] - rt[3]) < eps:
                        # x1 = x2 != x3 = x4
                        self.ordered_z = rt
                        self.type_no = 47
                    else:
                        # x1 = x2 != x3 != x4
                        self.ordered_z = rt
                        self.type_no = 48
            else:
                if abs(rt[1] - rt[2]) < eps:
                    if abs(rt[2] - rt[3]) < eps:
                        # x1 != x2 = x3 = x4
                        self.ordered_z = rt[[1, 2, 3, 0]]
                        self.type_no = 46
                    else:
                        # x1 != x2 = x3 != x4
                        self.ordered_z = rt[[1, 2, 0, 3]]
                        self.type_no = 48
                else:
                    if abs(rt[2] - rt[3]) < eps:
                        # x1 != x2 != x3 = x4
                        self.ordered_z = rt[[2, 3, 0, 1]]
                        self.type_no = 48
                    else:
                        # x1 != x2 != x3 != x4
                        self.ordered_z = rt
                        self.type_no = 49
