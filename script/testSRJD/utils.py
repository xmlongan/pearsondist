import numpy as np
from matplotlib import pyplot as plt

def plot(x, fx, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, fx, 'b-', lw=1, label='f(x)')
    plt.title(title)
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dpdf_pdf(x, dpdf, pdf, title=None, roots=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, dpdf, 'b-', lw=1, label='dpdf(x)')
    plt.plot(x, pdf, 'r-', lw=1, label='pdf(x)')

    if len(roots) == 2:
        plt.axvline(x = roots[0], color = 'r', linestyle='--', linewidth=0.5)
        plt.axvline(x = roots[1], color = 'r', linestyle='--', linewidth=0.5)
    if len(roots) == 4:
        roots = sorted(roots, key=abs)
        plt.axvline(x=roots[0], color='r', linestyle='--', linewidth=0.5)
        plt.axvline(x=roots[1], color='r', linestyle='--', linewidth=0.5)

    plt.title(title)
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pdf(x, pdf):
    # Plot the PDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, 'b-', lw=1, label='PDF')

    # Customize the plot
    plt.title('Distribution Density Function')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cdf(x, pdf):
    dx = pdf
    h = x[1] - x[0]
    # cumsum the middle ones except the first and last ones
    cum = np.cumsum(dx[1:(-1)] * h)
    #
    px = np.zeros(len(x))
    px[1] = (dx[0] + dx[1]) * h / 2
    px[2:] = (dx[0] + dx[2:]) * h / 2 + cum
    cdf = px

    plt.figure(figsize=(10, 6))
    plt.plot(x, cdf, 'b-', lw=1, label='CDF')
    # Customize the plot
    plt.title('Cumulative Distribution Function')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pdf_cdf(x, pdf, title=None):
    dx = pdf
    h = x[1] - x[0]
    # cumsum the middle ones except the first and last ones
    cum = np.cumsum(dx[1:(-1)] * h)
    #
    px = np.zeros(len(x))
    px[1] = (dx[0] + dx[1]) * h / 2
    px[2:] = (dx[0] + dx[2:]) * h / 2 + cum
    # cdf = px
    cdf = px / px[-1]  # normalize to 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    ax1.plot(x, pdf, 'g-', lw=1, label='PDF')
    ax1.set_title('Distribution Density Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, cdf, 'b-', lw=1, label='CDF')
    ax2.set_title('Cumulative Distribution Function')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
