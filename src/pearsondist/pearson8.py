"""
I defined a class :py:class:`Pearson8` to construct Pearson distributions that
match the first eight moments of the unknown distributions.
"""
import numpy as np

from pearsondist.adjust_lb_ub import adjust_lb_ub
from pearsondist.pfdecom4 import PFDecom4
from pearsondist.support8 import Support8
from pearsondist.pdf import Pdf


class Pearson8:
    """Class for Pearson distributions matching the first eight moments"""

    mom: list = None     # the first eight moments
    """the first eight moments"""
    coef: list = None    # a, c0, c1, c2, c3, c4
    """coefficients of the Pearson distribution"""
    pfd: dict = None     # partial fraction decomposition
    """Partial Fraction Decomposition of the Pearson distribution"""

    pdf_obj: Pdf = None
    """Un-normalized PDF of the Pearson distribution"""

    lower_bound: float = None
    """The lower bound of the support of the distribution"""
    upper_bound: float = None
    """The upper bound of the support of the distribution"""
    bounds: tuple = None
    """(lower bound, upper bound)"""

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
        self.pdf_obj = Pdf(self.pfd, self.coef)
        # print(f'isMax: {self.pdf_obj.is_max}')

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
        return self.pdf_obj.pdf(x)

    def dpdf(self, x):
        """Derivative of the Pearson density function"""
        return self.pdf_obj.dpdf(x)

    def determine_bounds(self):
        support8 = Support8(self.pdf_obj)
        return adjust_lb_ub(support8.lower_bound, support8.upper_bound, self.pfd)
