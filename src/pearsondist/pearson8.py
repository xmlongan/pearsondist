"""
I defined a class :py:class:`Pearson8` to construct Pearson distributions that
match the first eight moments of the unknown distributions.
"""
import numpy as np
from pearsondist.pfdecom4 import PFDecom4


class Pearson8:
    """Class for Pearson distributions matching the first eight moments"""

    mom: list = None     # the first eight moments
    """the first eight moments"""
    coef: list = None    # a, c0, c1, c2, c3, c4
    """coefficients of the Pearson distribution"""
    pfd: dict = None     # partial fraction decomposition
    """Partial Fraction Decomposition of the Pearson distribution"""
    scale: float = None
    """scale of the Pearson density function
    
    Scale up|down the density, such that the maximum density (at x = -a) equal 1.
    Actually, it may be the minimum density instead of the maximum one equal 1.
    """

    def __init__(self, moment: list):
        r"""Initialize Pearson8 object

        :param list moment: the first eight or more raw moments, noting that :math:`\mu_0`
          should not be included, and moments with order larger than eight will be ignored.
        """
        if len(moment) < 8:
            raise ValueError('mom_to_coef expects at least 8 moments')
        self.mom = moment[:8].copy()
        self.mom_to_coef()
        pfdecomp = PFDecom4(self.coef)
        self.pfd = pfdecomp.pfd
        a = self.coef[0]
        if self.pfd['type'] == 41: self.scale = self.log_pdf81(-a)
        if self.pfd['type'] == 42: self.scale = self.log_pdf82(-a)
        if self.pfd['type'] == 43: self.scale = self.log_pdf83(-a)
        if self.pfd['type'] == 44: self.scale = self.log_pdf84(-a)
        if self.pfd['type'] == 45: self.scale = self.log_pdf85(-a)
        if self.pfd['type'] == 46: self.scale = self.log_pdf86(-a)
        if self.pfd['type'] == 47: self.scale = self.log_pdf87(-a)
        if self.pfd['type'] == 48: self.scale = self.log_pdf88(-a)
        if self.pfd['type'] == 49: self.scale = self.log_pdf89(-a)

    def mom_to_coef(self):
        """From moments to coefficients

        Given the first eight moments, how to get the coefficients in the partial
        differential equation that satisfied by the Pearson density function.
        :return: None
        """
        mom = self.mom
        m1, m2, m3, m4 = mom[0], mom[1], mom[2], mom[3]
        m5, m6, m7, m8 = mom[4], mom[5], mom[6], mom[7]
        a = np.array([
            [1, 0, -1, -2 * m1, -3 * m2, -4 * m3],
            [m1, -1, -2 * m1, -3 * m2, -4 * m3, -5 * m4],
            [m2, -2 * m1, -3 * m2, -4 * m3, -5 * m4, -6 * m5],
            [m3, -3 * m2, -4 * m3, -5 * m4, -6 * m5, -7 * m6],
            [m4, -4 * m3, -5 * m4, -6 * m5, -7 * m6, -8 * m7],
            [m5, -5 * m4, -6 * m5, -7 * m6, -8 * m7, -9 * m8]
        ])
        b = np.array(mom[0:6])
        x = np.linalg.solve(a, -b)  # solve ax = -b
        self.coef = list(x)

    def pdf(self, x):
        """Probability density function

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        if self.pfd['type'] == 41: return np.exp(self.log_pdf81(x) - self.scale)
        if self.pfd['type'] == 42: return np.exp(self.log_pdf82(x) - self.scale)
        if self.pfd['type'] == 43: return np.exp(self.log_pdf83(x) - self.scale)
        if self.pfd['type'] == 44: return np.exp(self.log_pdf84(x) - self.scale)
        if self.pfd['type'] == 45: return np.exp(self.log_pdf85(x) - self.scale)
        if self.pfd['type'] == 46: return np.exp(self.log_pdf86(x) - self.scale)
        if self.pfd['type'] == 47: return np.exp(self.log_pdf87(x) - self.scale)
        if self.pfd['type'] == 48: return np.exp(self.log_pdf88(x) - self.scale)
        if self.pfd['type'] == 49: return np.exp(self.log_pdf89(x) - self.scale)

    def log_pdf81(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 41

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        A, B = pfd['A1'], pfd['B1']
        t, m, B = x - pfd['x1'].real, pfd['x1'].imag, B + A * pfd['x1'].real
        pow = A / (2 * (t ** 2 + m ** 2)) - (B / (2 * m ** 2)) * (t / (t ** 2 + m ** 2))
        pow -= (B / (2 * m ** 3)) * np.atan(t / m)
        return pow

    def log_pdf82(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 42

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1, B1 = pfd['x1'], pfd['A1'], pfd['B1']
        x2, A2, B2 = pfd['x2'], pfd['A2'], pfd['B2']
        t1, m1 = x - x1.real, x1.imag
        t2, m2 = x - x2.real, x2.imag
        pow1 = -A1 * np.log(t1 ** 2 + m1 ** 2) / 2 - ((B1 + A1 * x1.real) / m1) * np.atan(t1 / m1)
        pow2 = -A2 * np.log(t2 ** 2 + m2 ** 2) / 2 - ((B2 + A2 * x2.real) / m2) * np.atan(t2 / m2)
        return pow1 + pow2

    def log_pdf83(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 43

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1, A2 = pfd['x1'], pfd['A1'], pfd['A2']
        x3, A3, B3 = pfd['x3'], pfd['A3'], pfd['B3']
        t3, m3 = x - x3.real, x3.imag
        pow1 = -A1 * np.log(np.abs(x - x1)) + A2 / (x - x1)
        pow2 = -A3 * np.log(t3 ** 2 + m3 ** 2) / 2 - ((B3 + A3 * x3.real) / m3) * np.atan(t3 / m3)
        return pow1 + pow2

    def log_pdf84(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 44

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1 = pfd['x1'], pfd['A1']
        x2, A2 = pfd['x2'], pfd['A2']
        x3, A3, B3 = pfd['x3'], pfd['A3'], pfd['B3']
        t3, m3 = x - x3.real, x3.imag
        pow1 = -A1 * np.log(np.abs(x - x1)) - A2 * np.log(np.abs(x - x2))
        pow2 = -A3 * np.log(t3 ** 2 + m3 ** 2) / 2 - ((B3 + A3 * x3.real) / m3) * np.atan(t3 / m3)
        # return np.exp(pow1 + pow2 + 3430)
        return pow1 + pow2

    def log_pdf85(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 45

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A3, A4 = pfd['x1'], pfd['A3'], pfd['A4']
        pow = A3 / (2 * (x - x1) ** 2) + A4 / (3 * (x - x1) ** 3)
        return pow

    def log_pdf86(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 46

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1, A2, A3 = pfd['x1'], pfd['A1'], pfd['A2'], pfd['A3']
        x4, A4 = pfd['x4'], pfd['A4']
        pow1 = -A1 * np.log(np.abs(x - x1)) + A2 / (x - x1) + A3 / (2 * (x - x1) ** 2)
        pow2 = -A4 * np.log(np.abs(x - x4))
        return pow1 + pow2

    def log_pdf87(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 47

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1, A2 = pfd['x1'], pfd['A1'], pfd['A2']
        x3, A3, A4 = pfd['x3'], pfd['A3'], pfd['A4']
        pow1 = -A1 * np.log(np.abs(x - x1)) + A2 / (x - x1)
        pow2 = -A3 * np.log(np.abs(x - x3)) + A4 / (x - x3)
        return pow1 + pow2

    def log_pdf88(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 48

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1, A2 = pfd['x1'], pfd['A1'], pfd['A2']
        x3, A3 = pfd['x3'], pfd['A3']
        x4, A4 = pfd['x4'], pfd['A4']
        pow1 = -A1 * np.log(np.abs(x - x1)) + A2 / (x - x1)
        pow2 = -A3 * np.log(np.abs(x - x3)) - A4 * np.log(np.abs(x - x4))
        return pow1 + pow2

    def log_pdf89(self, x):
        """log density function when :abbr:`PFD(Partial Fraction Decomposition)` type is 49

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: density function value
        :rtype: np.float or np.array
        """
        pfd = self.pfd
        x1, A1 = pfd['x1'], pfd['A1']
        x2, A2 = pfd['x2'], pfd['A2']
        x3, A3 = pfd['x3'], pfd['A3']
        x4, A4 = pfd['x4'], pfd['A4']
        pow1 = -A1 * np.log(np.abs(x - x1)) - A2 * np.log(np.abs(x - x2))
        pow2 = -A3 * np.log(np.abs(x - x3)) - A4 * np.log(np.abs(x - x4))
        return pow1 + pow2
