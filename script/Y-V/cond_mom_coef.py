import numpy as np


def compute_conditional_moment_coeffs(joint_moments, v_moments, max_order_v=4):
    """
    joint_moments: 字典或数组，存储 E[Y^k * V^j]
    v_moments: 数组，存储 E[V^j]
    max_order_v: V 的多项式拟合阶数 (通常 4-6 阶已足够)
    """
    # 构建 Hankel 矩阵 M_V (形状: (max_order_v+1, max_order_v+1))
    M_V = np.array([[v_moments[i + j] for j in range(max_order_v + 1)]
                    for i in range(max_order_v + 1)])

    coeffs_matrix = {}
    for k in range(1, 9):  # 需要 Y 的前 8 阶条件矩
        # 构建常数向量 C_k (形状: (max_order_v+1, 1))
        C_k = np.array([joint_moments[k][j] for j in range(max_order_v + 1)])

        # 求解 A_k = M_V^-1 * C_k
        # 使用 lstsq 以增加数值稳定性
        A_k, residuals, rank, s = np.linalg.lstsq(M_V, C_k, rcond=None)
        coeffs_matrix[k] = A_k

    return coeffs_matrix
