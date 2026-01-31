import math

import numpy as np
import warnings
from pearsondist.pdf import Pdf

class Support8:
    """Class for determining the support of the Pearson distribution"""

    lower_bound: float = None
    """The lower bound of the support of the distribution"""
    upper_bound: float = None
    """The upper bound of the support of the distribution"""

    coef: list = None
    """coefficients: a, c0, c1, c2, c3, c4"""
    pdf_obj: Pdf = None
    """Un-normalized PDF of the Pearson distribution"""
    def __init__(self, pdf_obj):
        self.pdf_obj = pdf_obj
        self.coef = pdf_obj.coef
        self.lower_bound, self.upper_bound = self.determine_bounds()

    def determine_bounds(self):
        # special case: isMin
        if not self.pdf_obj.is_max:
            return self.u_bounds()
        # special case: isMax and type 49
        elif self.pdf_obj.pfd['type'] == 49:
            return self.distinct4roots_bounds()
        # typical cases: isMax and not type 49
        a = self.coef[0] # -a argmax_pdf
        argmax_dpdf, argmin_dpdf = self.pdf_obj.arg_max_min_dpdf()
        # determine lower bound
        distance1 = -a - argmax_dpdf
        x0_left = argmax_dpdf - distance1
        lb = x0_left - 10 * distance1; ub = argmax_dpdf
        lower_bound = self.newton(x0_left, lb, ub)
        # determine upper bound
        distance2 = argmin_dpdf - (-a)
        x0_right = argmin_dpdf + distance2
        lb = argmin_dpdf; ub = x0_right + 10 * distance2
        upper_bound = self.newton(x0_right, lb, ub)
        return lower_bound, upper_bound

    def newton(self, x0, lb, ub, eps=1e-5, iter_max=10):
        """solve pdf = 0"""
        iteration = 0
        msg = f'iteration: {iteration}, x0 = {x0:>12.7f}'
        while iteration < iter_max:
            x = x0 - self.pdf_obj.pdf_over_dpdf(x0)
            if abs(x - x0) < eps: break
            x0 = x
            iteration += 1
            if x0 > ub:
                x0 = ub; break
            elif x0 < lb:
                x0 = lb; break
            msg += f'\niteration: {iteration}, x0 = {x0:>12.7f}'
        print(msg)
        return x0

    def newton2(self, x0, lb, ub, eps=1e-5, iter_max=10):
        """solve dpdf = 0"""
        iteration = 0
        msg = f'iteration: {iteration}, x0 = {x0:>12.7f}'
        while iteration < iter_max:
            x = x0 - self.pdf_obj.dpdf_over_ddpdf(x0)
            if abs(x - x0) < eps: break
            x0 = x
            iteration += 1
            if x0 > ub:
                x0 = ub
                break
            elif x0 < lb:
                x0 = lb
                break
            msg += f'\niteration: {iteration}, x0 = {x0:>12.7f}'
        print(msg)
        return x0

    def u_bounds(self):
        """Cases where -a is argmin of the PDF

        the support must include 0, and should also include -a
        """
        if self.pdf_obj.is_max:
            raise NotImplementedError('-a is argmax, it should not be U-type.')
        pfd = self.pdf_obj.pfd
        type = pfd['type']
        a = self.coef[0]
        if type in [41, 42]:
            # 41: all complex: (x1, x2) = (x3, x4)
            # 42: all complex: (x1, x2) != (x3, x4)
            raise NotImplementedError('not implemented for type 41 and 42')
        elif type in [43, 45]:
            # 43: 2 real, 2 complex: x1 = x2, (x3, x4)
            # 45: 4 real: x1 = x2 = x3 = x4
            lb, ub = self.distinct1root_bounds()
        elif type in [44, 46, 47]:
            # 44: 2 real, 2 complex: x1 != x2, (x3, x4)
            # 46: 4 real: x1 = x2 = x3 != x4
            # 47: 4 real: x1 = x2 != x3 = x4
            lb, ub = self.distinct2roots_bounds()
        elif type == 48:
            # 48: 4 real: x1 = x2 != x3 != x4
            lb, ub = self.distinct3roots_bounds()
        elif type == 49:
            # 4 real: x1 != x2 != x3 != x4
            lb, ub = self.distinct4roots_bounds()
        else:
            raise ValueError(f"unknown root type: {type}")
        return lb, ub

    def distinct1root_bounds(self):
        # 43: 2 real, 2 complex: x1 = x2, (x3, x4)
        # 45: 4 real: x1 = x2 = x3 = x4
        pfd = self.pdf_obj.pfd
        x1 = pfd['x1']
        if 0 == x1:
            raise NotImplementedError('type 43|45 one root = 0, which should never happen')
        if x1 < 0:
            lb = x1 + 1e-7
            ub = math.inf
        else:
            lb = -math.inf
            ub = x1 - 1e-7
        self.effective_check(lb, ub)
        return lb, ub

    def distinct2roots_bounds(self):
        # 44: 2 real, 2 complex: x1 != x2, (x3, x4)
        # 46: 4 real: x1 = x2 = x3 != x4
        # 47: 4 real: x1 = x2 != x3 = x4
        pfd = self.pdf_obj.pfd
        if type == 44:
            x1 = pfd['x1']
            x2 = pfd['x2']
        elif type == 46:
            x1 = pfd['x1']
            x2 = pfd['x4']
            if x1 > x2: x1, x2 = x2, x1
        else:
            x1 = pfd['x1']
            x2 = pfd['x3']
            if x1 > x2: x1, x2 = x2, x1
        if 0 in [x1, x2]:
            raise NotImplementedError('type 44|46|47 one root = 0, which should never happen')
        # x1 < x2
        if 0 < x1:
            lb = -math.inf
            ub = x1 - 1e-7
        elif x2 < 0:
            lb = x2 + 1e-7
            ub = math.inf
        else:
            # x1 < 0 < x2
            lb = x1 + 1e-7
            ub = x2 - 1e-7
        self.effective_check(lb, ub)
        return lb, ub

    def distinct3roots_bounds(self):
        pfd = self.pdf_obj.pfd
        # 48: 4 real: x1 = x2 != x3 != x4
        x1 = pfd['x1']
        x2 = pfd['x3']
        x3 = pfd['x4']
        x = sorted([x1, x2, x3])
        if 0 in x:
            raise NotImplementedError('type 48 one root = 0, which should never happen')
        if 0 < x[0]:
            lb = -math.inf
            ub = x[0] - 1e-7
        elif x[2] < 0:
            lb = x[2] + 1e-7
            ub = math.inf
        else:
            # x1 < 0 < x3
            if 0 < x[1]:
                lb = x[0] + 1e-7
                ub = x[1] - 1e-7
            else:
                lb = x[1] + 1e-7
                ub = x[2] - 1e-7
        self.effective_check(lb, ub)
        return lb, ub

    def distinct4roots_bounds(self):
        # 49: 4 real: x1 != x2 != x3 != x4
        pfd = self.pdf_obj.pfd
        x1 = pfd['x1']
        x2 = pfd['x2']
        x3 = pfd['x3']
        x4 = pfd['x4']
        x = sorted([x1, x2, x3, x4])
        if 0 in x:
            raise NotImplementedError('type 49 one root = 0, which should never happen')
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        if 0 < x1:
            lb = -math.inf
            ub = x1 - 1e-7
        elif x4 < 0:
            lb = x4 + 1e-7
            ub = math.inf
        else:
            # x1 < 0 < x4
            if 0 < x2:
                lb = x1 + 1e-7
                ub = x2 - 1e-7
            elif 0 < x3:
                lb = x2 + 1e-7
                ub = x3 - 1e-7
            else:
                lb = x3 + 1e-7
                ub = x4 - 1e-7
        self.effective_check(lb, ub)
        return lb, ub

    def effective_check(self, lb, ub):
        a = self.coef[0]
        if -a < lb:
            warnings.warn(f'-a (={-a}) < lb ({lb})')
        if -a > ub:
            warnings.warn(f'-a (={-a}) > ub ({ub})')
        if 0 < lb or 0 > ub:
            raise Exception(f'(lb, ub) = ({lb}, {ub}) is not valid, 0 is not included.')