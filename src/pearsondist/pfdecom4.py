import numpy as np
from pearsondist.rootcatalog4 import RootCatalog4


class PFDecom4:

    coef: list = None
    pfd: dict = None

    def __init__(self, coef):
        if len(coef) != 6:
            raise ValueError('coef expects a, c0, c1, c2, c3, c4')
        self.coef = coef
        z = np.roots(list(reversed(coef[1:])))  # note: c4, c3, c2, c1, c0

        # print("c: ", list(reversed(coef[1:])))
        # print("z: ", z)

        z = RootCatalog4(z)
        if z.type_no == 41: self.pfd = self.pfd41(z.ordered_z)
        if z.type_no == 42: self.pfd = self.pfd42(z.ordered_z)
        if z.type_no == 43: self.pfd = self.pfd43(z.ordered_z)
        if z.type_no == 44: self.pfd = self.pfd44(z.ordered_z)
        if z.type_no == 45: self.pfd = self.pfd45(z.ordered_z)
        if z.type_no == 46: self.pfd = self.pfd46(z.ordered_z)
        if z.type_no == 47: self.pfd = self.pfd47(z.ordered_z)
        if z.type_no == 48: self.pfd = self.pfd48(z.ordered_z)
        if z.type_no == 49: self.pfd = self.pfd49(z.ordered_z)

    def pfd41(self, z):
        # all complex: (x1, x2) = (x3, x4)
        pfd = {'x1': z[0], 'A1': 1 / self.coef[-1], 'B1': self.coef[0] / self.coef[-1],
               'type': 41}
        return pfd

    def pfd42(self, z):
        # all complex: (x1, x2) != (x3, x4)
        p1, q1 = -2 * z[0].real, (np.abs(z[0])) ** 2  # work both for real and complex
        p2, q2 = -2 * z[2].real, (np.abs(z[2])) ** 2
        a = np.array([
            [1, 0, 1, 0],
            [p2, 1, p1, 1],
            [q2, p2, q1, p1],
            [0, q2, 0, q1]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1b1_a2b2 = np.linalg.solve(a, b)
        pfd = {'x1': z[0], 'A1': a1b1_a2b2[0], 'B1': a1b1_a2b2[1],
               'x2': z[2], 'A2': a1b1_a2b2[2], 'B2': a1b1_a2b2[3],
               'type': 42}
        return pfd

    def pfd43(self, z):
        # 2 real, 2 complex: x1 = x2, (x3, x4)
        x1 = z[0].real
        p, q = -2 * z[2].real, (np.abs(z[2])) ** 2
        a = np.array([
            [1, 0, 1, 0],
            [p - x1, 1, -2 * x1, 1],
            [q - x1 * p, p, x1 ** 2, -2 * x1],
            [-x1 * q, q, 0, x1 ** 2]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2_a3b3 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2_a3b3[0], 'A2': a1a2_a3b3[1],
               'x3': z[2], 'A3': a1a2_a3b3[2], 'B3': a1a2_a3b3[3],
               'type': 43}
        return pfd

    def pfd44(self, z):
        # 2 real, 2 complex: x1 != x2, (x3, x4)
        x1, x2, x3 = z[0].real, z[1].real, z[2]
        p, q = -2 * x3.real, (np.abs(x3)) ** 2
        a = np.array([
            [1, 1, 1, 0],
            [p - x2, p - x1, -(x1 + x2), 1],
            [q - x2 * p, q - x1 * p, x1 * x2, -(x1 + x2)],
            [-x2 * q, -x1 * q, 0, x1 * x2]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2_a3b3 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2_a3b3[0],
               'x2': x2, 'A2': a1a2_a3b3[1],
               'x3': x3, 'A3': a1a2_a3b3[2], 'B3': a1a2_a3b3[3],
               'type': 44}
        return pfd

    def pfd45(self, z):
        # 4 real: x1 = x2 = x3 = x4
        x1 = z[0].real
        pfd = {'x1': x1, 'A3': 1 / self.coef[-1], 'A4': self.coef[0] / self.coef[-1] + x1 / self.coef[-1],
               'type': 45}
        return pfd

    def pfd46(self, z):
        # 4 real: x1 = x2 = x3 != x4
        x1, x4 = z[0].real, z[3].real
        a = np.array([
            [1, 0, 0, 1],
            [-(2 * x1 + x4), 1, 0, -3 * x1],
            [x1 ** 2 + 2 * x1 * x4, -2 * (x1 + x4), 1, 3 * x1 ** 2],
            [-x1 ** 2 * x4, x1 * x4, -x4, -x1 ** 3]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2a3a4 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2a3a4[0], 'A2': a1a2a3a4[1], 'A3': a1a2a3a4[2],
               'x4': x4, 'A4': a1a2a3a4[3],
               'type': 46}
        return pfd

    def pfd47(self, z):
        # 4 real: x1 = x2 != x3 = x4
        x1, x3 = z[0].real, z[2].real
        a = np.array([
            [1, 0, 1, 0],
            [-(x1 + 2 * x3), 1, -(x3 + 2 * x1), 1],
            [2 * x1 * x3 + x3 ** 2, -2 * x3, 2 * x3 * x1 + x1 ** 2, -2 * x1],
            [-x1 * x3 ** 2, x3 ** 2, -x3 * x1 ** 2, x1 ** 2]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2a3a4 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2a3a4[0], 'A2': a1a2a3a4[1],
               'x3': x3, 'A3': a1a2a3a4[2], 'A4': a1a2a3a4[3],
               'type': 47}
        return pfd

    def pfd48(self, z):
        # 4 real: x1 = x2 != x3 != x4
        x1, x3, x4 = z[0].real, z[2].real, z[3].real
        a = np.array([
            [1, 0, 1, 1],
            [-(x1 + x3 + x4), 1, -(2 * x1 + x4), -(2 * x1 + x3)],
            [x3 * x4 + x1 * (x3 + x4), -(x3 + x4), x1 ** 2 + 2 * x1 * x4, x1 ** 2 + 2 * x1 * x3],
            [-x1 * x3 * x4, x3 * x4, -x1 ** 2 * x4, -x1 ** 2 * x3]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2a3a4 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2a3a4[0], 'A2': a1a2a3a4[1],
               'x3': x3, 'A3': a1a2a3a4[2],
               'x4': x4, 'A4': a1a2a3a4[3],
               'type': 48}
        return pfd

    def pfd49(self, z):
        # 4 real: x1 != x2 != x3 != x4
        x1, x2, x3, x4 = z[0].real, z[1].real, z[2].real, z[3].real
        a = np.array([
            [1, 1, 1, 1],
            [-(x2 + x3 + x4), -(x1 + x3 + x4), -(x1 + x2 + x4), -(x1 + x2 + x3)],
            [x3 * x4 + x2 * (x3 + x4), x3 * x4 + x1 * (x3 + x4), x1 * x2 + x4 * (x1 + x2), x1 * x2 + x3 * (x1 + x2)],
            [-x2 * x3 * x4, -x1 * x3 * x4, -x1 * x2 * x4, -x1 * x2 * x3]
        ])
        b = np.array([0, 0, 1 / self.coef[-1], self.coef[0] / self.coef[-1]])
        a1a2a3a4 = np.linalg.solve(a, b)
        pfd = {'x1': x1, 'A1': a1a2a3a4[0],
               'x2': x2, 'A2': a1a2a3a4[1],
               'x3': x3, 'A3': a1a2a3a4[2],
               'x4': x4, 'A4': a1a2a3a4[3],
               'type': 49}
        return pfd
