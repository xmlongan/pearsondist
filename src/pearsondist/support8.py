import numpy as np
import warnings

class Support8:

    lower_bound: float = None
    upper_bound: float = None
    coef: list = None

    argmax_dpdf: float = None
    argmin_dpdf: float = None

    def __init__(self, coef):
        if len(coef) != 6:
            raise ValueError('coef expects a, c0, c1, c2, c3, c4')
        self.coef = coef
        self.arg_max_min_dpdf()

    def arg_max_min_dpdf(self):
        r"""Get argmax and argmin of the derivative of the PDF.

        The second derivative of the density function is given by
        :math:`\frac{d p^2(x)}{dx^2} = \frac{P(x)}{Q(x)} p(x)` where
        :math:`p(x)` denotes the density function,
        :math:`P(x) = 3c_4 x^4 + (2c_3 + 4c_4a) x^3 + (c_2 + 3c_3a + 1) x^2 + 2a(c_2+1) x + (a^2 + c_1a - c_0)`,
        :math:`Q(x) = (c_0 + c_1x + c_2x^2 + c_3x^3 + c_4x^4)^2`.
        Therefore, to find roots of this second derivative is equivalent to
        finding roots of :math:`P(x)`, i.e., solving :math:`P(x) = 0`.

        Note that dpdf(-a) = 0
            argmax_dpdf < -a
            argmin_dpdf > -a
        """
        a, c0, c1 = self.coef[0], self.coef[1], self.coef[2]
        c2, c3, c4 = self.coef[3], self.coef[4], self.coef[5]
        # The coefficients are ordered from the highest power to lowest (x^4 to x^0)
        coef_ddpdf = [3*c4, 2*c3 + 4*c4*a, c2 + 3*c3*a + 1, 2*a*(c2 + 1), a**2 + c1*a - c0]
        roots = np.roots(coef_ddpdf)
        print(f"The roots are: {roots}")
        real_roots = np.sort(roots[np.isreal(roots)].real)
        #
        # Find the index of the first element larger than `-a` (dpdf = 0)
        idx = np.searchsorted(real_roots, -a, side='right')
        if idx < len(real_roots):
            self.argmin_dpdf = real_roots[idx]
        else:
            warnings.warn("no ddpdf roots larger than -a")
        # Find the index of the first element not less than `val`
        idx = np.searchsorted(real_roots, -a, side='left')
        if idx > 0:
            self.argmax_dpdf = real_roots[idx - 1]
        else:
            warnings.warn("no ddpdf roots less than -a")

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
        num = (a+x) * (c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4)
        den = ((3*c4)*x**4
               + (2*c3 + 4*c4*a)*x**3
               + (c2 + 3*c3*a + 1)*x**2
               + 2*a*(c2 + 1)*x
               + (a**2 + c1*a - c0))
        return - num/den

    def newton(self, x0, eps=1e-7, iter_max=10):
        iteration = 0
        while iteration < iter_max:
            x = x0 - self.dpdf_over_ddpdf(x0)
            if abs(x - x0) < eps: break
            x0 = x
            iteration += 1
        return x0

    def determine_bounds(self):
        self.lower_bound = self.newton(self.argmax_dpdf - 1e-5)
        self.upper_bound = self.newton(self.argmin_dpdf + 1e-5)
