import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gamma


def get_pure_grid_pdf(samples, grid_size=7000):
    samples = np.asarray(samples)
    min_v, max_v = np.min(samples), np.max(samples)
    grid = np.linspace(min_v, max_v, grid_size)
    dx = grid[1] - grid[0]

    # 线性分桶（统计意义上的稳健分配）
    counts = np.zeros(grid_size)
    indices = (samples - min_v) / dx

    idx_l = np.floor(indices).astype(int)
    idx_r = idx_l + 1
    w_r = indices - idx_l

    # 仅保留有效索引
    mask = (idx_l >= 0) & (idx_r < grid_size)
    np.add.at(counts, idx_l[mask], 1.0 - w_r[mask])
    np.add.at(counts, idx_r[mask], w_r[mask])

    # 归一化获得 PDF 向量
    pdf_vector = counts / (len(samples) * dx)

    return grid, pdf_vector  # 返回的是两个固定的数组，不含任何插值


class EmpiricalPDF:
    def __init__(self, samples, grid_size=7000):
        self.samples = np.asarray(samples)
        self.min_val = np.min(self.samples)
        self.max_val = np.max(self.samples)

        # 创建均匀网格
        self.grid = np.linspace(self.min_val, self.max_val, grid_size)
        self.dx = self.grid[1] - self.grid[0]

        # 线性分桶逻辑
        indices = (self.samples - self.min_val) / self.dx
        idx_left = np.floor(indices).astype(int)
        idx_right = idx_left + 1
        weight_right = indices - idx_left
        weight_left = 1.0 - weight_right

        valid_mask = (idx_left >= 0) & (idx_right < grid_size)
        idx_left = idx_left[valid_mask]
        idx_right = idx_right[valid_mask]
        weight_left = weight_left[valid_mask]
        weight_right = weight_right[valid_mask]

        pdf_values = np.zeros(grid_size)
        np.add.at(pdf_values, idx_left, weight_left)
        np.add.at(pdf_values, idx_right, weight_right)

        self.pdf_values = pdf_values / (len(self.samples) * self.dx)

        self._interpolator = interp1d(
            self.grid, self.pdf_values,
            kind='linear', bounds_error=False, fill_value=0.0
        )
        # kind='linear'

    def __call__(self, x):
        # return self._interpolator(x)
        val = self._interpolator(x)
        return np.maximum(val, 0.0)  # 强制保证概率密度非负


# --- 绘图与对比部分 ---
def plot_comparison():
    # 1. 生成模拟 CIR 过程特征的样本 (Gamma 分布)
    # 这里的参数 a=2, scale=0.5 模拟了一个左偏且有长尾的分布
    shape_param = 2.0
    scale_param = 0.5
    n_samples = 2560000
    grid_size = 100

    print(f"正在生成 {n_samples} 个 Gamma 分布样本...")
    data = np.random.gamma(shape_param, scale_param, size=n_samples)

    # 2. 构建经验 PDF
    print(f"正在构建经验 PDF (gridSize={grid_size})...")
    get_pdf = EmpiricalPDF(data, grid_size=grid_size)

    # 3. 准备绘图数据
    x_plot = np.linspace(0, np.percentile(data, 99.5), 1000)  # 绘制到 99.5% 分位数处，避免长尾拉得太长
    y_empirical = get_pdf(x_plot)
    y_theoretical = gamma.pdf(x_plot, shape_param, scale=scale_param)  # 理论 PDF

    # 4. 开始绘图
    plt.figure(figsize=(10, 6))

    # 绘制原始样本的直方图作为背景（使用 100 个 Bin 即可，用于观察大轮廓）
    plt.hist(data, bins=100, density=True, alpha=0.2, color='gray', label='Sample Histogram')

    # 绘制经验 PDF (我们的算法结果)
    plt.plot(x_plot, y_empirical, 'r-', linewidth=1, label='Empirical PDF (Linear Binning)')

    # 绘制理论 PDF (作为基准)
    plt.plot(x_plot, y_theoretical, 'b--', linewidth=1, label='True Theoretical PDF')

    plt.title(f"Empirical PDF Approximation (n={n_samples}, gridSize={grid_size})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # 局部放大展示平滑度
    plt.axes([0.5, 0.3, 0.35, 0.35])
    zoom_x = np.linspace(1.0, 1.5, 100)
    plt.plot(zoom_x, gamma.pdf(zoom_x, shape_param, scale=scale_param), 'b--')
    plt.plot(zoom_x, get_pdf(zoom_x), 'r-')
    plt.title("Local Zoom-in")

    plt.show()


if __name__ == "__main__":
    plot_comparison()