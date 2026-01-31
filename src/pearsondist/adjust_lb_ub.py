"""
Adjust the lower and upper bound of the Pearson distribution
according to the Partial Fraction Decomposition, i.e.,
the type of the roots.
"""

def adjust_lb_ub(lb, ub, pfd):
    # assume the support lb < 0 < ub
    if lb >= 0.0 or ub <= 0.0:
        raise ValueError(f'lb({lb:.7f}) >= 0.0 or ub({ub:.7f}) <= 0.0')
    lbub = [lb, ub]
    type_pfd = pfd['type']
    if type_pfd == 43 or type_pfd == 45:
        # two real: x1 = x2 or 4 real: x1 = x2 = x3 = x4
        x1 = pfd['x1']
        if 0.0 <= x1:
            if ub >= x1: lbub[1] = x1 - 1.0e-5
        else:
            if x1 >= lb: lbub[0] = x1 + 1.0e-5
    elif type_pfd == 44:
        # two real: x1 < x2
        x1 = pfd['x1']; x2 = pfd['x2']
        return adjust_lb_ub_2(lb, ub, x1, x2)
    elif type_pfd == 46:
        # x1 = x2 = x3 != x4 | x2 = x3 = x4 != x1
        x1 = pfd['x1']; x4 = pfd['x4']
        if x1 < x4:
            return adjust_lb_ub_2(lb, ub, x1, x4)
        else:
            return adjust_lb_ub_2(lb, ub, x4, x1)
    elif type_pfd == 47:
        # x1 = x2 < x3 = x4
        x1 = pfd['x1']; x3 = pfd['x3']
        return adjust_lb_ub_2(lb, ub, x1, x3)
    elif type_pfd == 48:
        # x1 = x2 != x3 != x4 | x2 = x3 != x1 != x4 | x3 = x4 != x1 != x2
        x1 = pfd['x1']; x3 = pfd['x3']; x4 = pfd['x4']
        x = sorted([x1, x3, x4])
        return adjust_lb_ub_3(lb, ub, x[0], x[1], x[2])
    elif type_pfd == 49:
        # x1 != x2 != x3 != x4
        x1 = pfd['x1']; x2 = pfd['x2']; x3 = pfd['x3']; x4 = pfd['x4']
        return adjust_lb_ub_4(lb, ub, x1, x2, x3, x4)
    return lbub


def trunc_lb_ub(lb, ub, x1, x2):
    # x1 < 0 < x2
    lbub = [lb, ub]
    if x1 >= lb: lbub[0] = x1 + 1.0e-5
    if x2 <= ub: lbub[1] = x2 - 1.0e-5
    return lbub


def adjust_lb_ub_2(lb, ub, x1, x2):
    # x1 < x2
    lbub = [lb, ub]
    if 0.0 <= x1:
        if ub >= x1: lbub[1] = x1 - 1.0e-5
    elif x2 <= 0.0:
        if x2 >= lb: lbub[0] = x2 + 1.0e-5
    else:
        return trunc_lb_ub(lb, ub, x1, x2)
    return lbub


def adjust_lb_ub_3(lb, ub, x1, x2, x3):
    # x1 < x2 < x3
    lbub = [lb, ub]
    if 0.0 <= x1:
        if ub >= x1: lbub[1] = x1 - 1.0e-5
    elif x3 <= 0.0:
        if x3 >= lb: lbub[0] = x3 + 1.0e-5
    else:
        if x2 < 0.0: return trunc_lb_ub(lb, ub, x2, x3)
        return trunc_lb_ub(lb, ub, x1, x2)
    return lbub


def adjust_lb_ub_4(lb, ub, x1, x2, x3, x4):
    # x1 < x2 < x3 < x4
    lbub = [lb, ub]
    if 0.0 <= x1:
        if ub >= x1: lbub[1] = x1 - 1.0e-5
    elif x4 <= 0.0:
        if x4 >= lb: lbub[0] = x4 + 1.0e-5
    else:
        # x1 < 0 < x4
        if 0.0 <= x2: return trunc_lb_ub(lb, ub, x1, x2)
        if x3 <= 0.0: return trunc_lb_ub(lb, ub, x3, x4)
        return trunc_lb_ub(lb, ub, x2, x3)
    return lbub
