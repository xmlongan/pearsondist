"""
Produce standardized moments from raw moments
"""
import math
import warnings


def stdmom(mom: list) -> tuple:
    m1, m2, m3, m4 = mom[0], mom[1], mom[2], mom[3]
    var = m2 - m1 ** 2
    std = math.sqrt(var)
    skewness = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / (std ** 3)
    kurtosis = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / (std ** 4)
    # check kurtosis bound and var > 0
    if var <= 0:
        raise ValueError(f'var = {var:.7f} <= 0')
    if std < 1e-5:
        warnings.warn(f'std < 1e-5, skewness and kurtosis may not be reliable!')
    if kurtosis < skewness ** 2 + 1:
        msg = f'kurtosis ({kurtosis:.7f}) < skewness ({skewness:.7f}) + 1'
        raise ValueError(msg)
    return m1, var, skewness, kurtosis
