import numpy as np
from scipy.special import genlaguerre, gamma
import matplotlib.pyplot as plt


class LaguerreDensity:
    def __init__(self, moments):
        """
        moments: List or array of raw moments [m1, m2, ..., m8]
        Note: m0 is assumed to be 1.0
        """
        self.m = np.insert(np.array(moments), 0, 1.0)  # [m0, m1, ..., m8]
        self.n_max = len(self.m) - 1

        # 1. 确定基础 Gamma 分布参数 (k, theta)
        m1, m2 = self.m[1], self.m[2]
        self.theta = (m2 - m1 ** 2) / m1
        self.k = m1 ** 2 / (m2 - m1 ** 2)
        self.alpha = self.k - 1

        # 2. 计算系数 a_n
        self.a = self._compute_coefficients()

    def _compute_coefficients(self):
        a_coeffs = np.zeros(self.n_max + 1)
        # a0 永远是 1, a1 和 a2 因为匹配了均值方差通常为 0
        for n in range(self.n_max + 1):
            # 获取第 n 阶广义拉盖尔多项式 L_n^(alpha)(x)
            # Ln 是一个 scipy 多项式对象
            Ln = genlaguerre(n, self.alpha)

            # 计算期望 E[Ln(V/theta)]
            # Ln.coeffs 是多项式系数 [c_n, c_{n-1}, ..., c_0] 对应 x^n, x^{n-1}...
            # 我们需要将其与归一化矩 m_k / theta^k 结合
            e_ln = 0
            poly_coeffs = Ln.coef[::-1]  # 反转变为 [c_0, c_1, ..., c_n]
            for k, ck in enumerate(poly_coeffs):
                e_ln += ck * (self.m[k] / (self.theta ** k))

            # 根据推导公式: a_n = (n! * Gamma(k)) / Gamma(n+k) * E[Ln]
            norm_fact = (gamma(n + 1) * gamma(self.k)) / gamma(n + self.k)
            a_coeffs[n] = norm_fact * e_ln

        return a_coeffs

    def pdf(self, v):
        """ 计算近似密度 f(v) """
        v = np.atleast_1d(v)
        x = v / self.theta

        # 基础 Gamma 密度 g(v)
        g_v = (v ** self.alpha * np.exp(-x)) / (self.theta ** self.k * gamma(self.k))

        # 级数修正项 (1 + a3*L3 + ... + a8*L8)
        correction = np.zeros_like(v)
        for n in range(self.n_max + 1):
            Ln_val = genlaguerre(n, self.alpha)(x)
            correction += self.a[n] * Ln_val

        return g_v * correction

