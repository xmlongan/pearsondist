"""
Probability Density Functions, unnormalized.
"""
import numpy as np
import warnings


class Pdf:
    """Class for unnormalized PDF of Pearson distribution"""

    pfd: dict = None  # partial fraction decomposition
    """Partial Fraction Decomposition of the Pearson distribution"""
    coef: list = None
    """coefficients: a, c0, c1, c2, c3, c4"""
    scale: float = None
    """scale of the Pearson density function
    PDF(-a), maximum or minimum of the PDF.

    Scale up|down the density, such that the maximum|minimum density (at x = -a) equals 1.
    """
    is_max: bool = None
    """whether -a is argmax or argmin"""

    def __init__(self, pfd, coef):
        flag = pfd['type'] in [41,42,43,44,45,46,47,48,49]
        if not flag:
            raise ValueError('Invalid pfd')
        if len(coef) != 6:
            raise ValueError('coef expects a, c0, c1, c2, c3, c4')
        self.pfd = pfd
        self.coef = coef
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
        self.is_max = self.isMax()

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
        return None

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

    def dpdf(self, x):
        """Derivative of the density function

        :param float x: input value of the density function, it should be within
          the support of the distribution.
        :return: derivative of density function.
        :rtype: np.float or np.array"""
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        num = a + x
        den = c0 + c1 * x + c2 * (x ** 2) + c3 * (x ** 3) + c4 * (x ** 4)
        return - (num / den) * self.pdf(x)

    def isMax(self):
        """whether the density function is max at -a"""
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        # The coefficients are ordered from the highest power to lowest (x^4 to x^0)
        coef = [3 * c4, 2 * c3 + 4 * c4 * a, c2 + 3 * c3 * a + 1, 2 * a * (c2 + 1), a ** 2 + c1 * a - c0]
        ddf = np.polyval(coef, -a)
        if ddf < 0:
            return True
        elif ddf > 0:
            return False
        else:
            warnings.warn("haven't figured out yet whether -a is the max or min!")
            return None

    def pdf_over_dpdf(self, x):
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        num = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
        den = a + x
        return -num/den

    def dpdf_over_ddpdf(self, x):
        r"""Ratio between the first and second derivative of the PDF.

        In order to find roots of the first derivative of the PDF.
        Note that the roots to find exclude the one :math:`x = -a`.

        :param float x: input value of the density function.
        :return: ratio at x.
        :rtype: float
        """
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        num = (a + x) * (c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4)
        den = ((3 * c4) * x ** 4
               + (2 * c3 + 4 * c4 * a) * x ** 3
               + (c2 + 3 * c3 * a + 1) * x ** 2
               + 2 * a * (c2 + 1) * x
               + (a ** 2 + c1 * a - c0))
        return - num / den

    def ddpdf_roots(self):
        r"""Get roots of the second derivative of the PDF.

        The second derivative of the density function is given by
        :math:`\frac{d p^2(x)}{dx^2} = \frac{P(x)}{Q(x)} p(x)` where
        :math:`p(x)` denotes the density function,
        :math:`P(x) = 3c_4 x^4 + (2c_3 + 4c_4a) x^3 + (c_2 + 3c_3a + 1) x^2 + 2a(c_2+1) x + (a^2 + c_1a - c_0)`,
        :math:`Q(x) = (c_0 + c_1x + c_2x^2 + c_3x^3 + c_4x^4)^2`.
        Therefore, to find roots of this second derivative is equivalent to
        finding roots of :math:`P(x)`, i.e., solving :math:`P(x) = 0`.
        """
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        # The coefficients are ordered from the highest power to lowest (x^4 to x^0)
        coef_ddpdf = [3*c4, 2*c3 + 4*c4*a, c2 + 3*c3*a + 1, 2*a*(c2 + 1), a**2 + c1*a - c0]
        roots = np.roots(coef_ddpdf)
        # print(f"The roots of ddpdf are: {roots}")
        return roots[np.isreal(roots)].real

    def arg_max_min_dpdf(self):
        r"""Get argmax and argmin of the derivative of the PDF.

        Limit to cases PDF(-a) reach the maximum value.

        From roots of the second derivative of the PDF.
        Note that dpdf(-a) = 0, argmax_dpdf < -a, argmin_dpdf > -a

        :return: argmax and argmin of the derivative of the PDF.
        :rtype: tuple
        """
        if not self.is_max:
            raise NotImplementedError('Not implemented for -a being the minimum')
        roots = self.ddpdf_roots()
        # sort
        real_roots = np.sort(roots[np.isreal(roots)].real)
        #
        a = self.coef[0]
        # Find the index of the first element larger than `-a` (dpdf = 0)
        idx = np.searchsorted(real_roots, -a, side='right')
        if idx < len(real_roots):
            argmin_dpdf = real_roots[idx]
        else:
            warnings.warn("no ddpdf roots larger than -a")
            argmin_dpdf = None
        # Find the index of the first element not less than `val`
        idx = np.searchsorted(real_roots, -a, side='left')
        if idx > 0:
            argmax_dpdf = real_roots[idx - 1]
        else:
            warnings.warn("no ddpdf roots less than -a")
            argmax_dpdf = None
        return argmax_dpdf, argmin_dpdf
