"""Grafico del q-esponenziale di Tsallis e_q(u) = [1 + (1-q) u]^(1/(1-q))."""
import math

import numpy as np
import matplotlib.pyplot as plt

# Palette categorica in ordine fisso (non ciclata)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300"]


def q_exp(u, q):
    """q-esponenziale e_q(u) = [1 + (1-q) u]^(1/(1-q)).

    Per q = 1 si riduce a exp(u). Dove 1 + (1-q) u <= 0 si usa il
    cutoff di Tsallis ([.]_+): vale 0 per q < 1, mentre per q > 1 la
    funzione diverge per u -> 1/(q-1) e oltre non è definita (NaN).
    """
    u = np.asarray(u, dtype=float)
    if np.isclose(q, 1.0):
        return np.exp(u)
    base = 1.0 + (1.0 - q) * u
    out = np.full_like(u, np.nan)
    pos = base > 0
    out[pos] = base[pos] ** (1.0 / (1.0 - q))
    if q < 1:
        out[~pos] = 0.0
    return out


def plot_q_exponential(q_values=(0.5, 1.0, 1.5, 2.0), u=None,
                       ymax=10.0, filename=None, show=True):
    """Traccia e_q(u) per ciascun q in q_values (massimo 6 curve)."""
    if len(q_values) > len(COLORS):
        raise ValueError(f"al massimo {len(COLORS)} valori di q")
    if u is None:
        u = np.linspace(-4, 3, 2001)

    fig, ax = plt.subplots(figsize=(8, 5))
    for q, color in zip(q_values, COLORS):
        y = q_exp(u, q)
        y = np.where(y > ymax * 1.5, np.nan, y)  # taglia la divergenza
        ax.plot(u, y, color=color, lw=2, label=f"q = {q:g}")

    ax.set_ylim(-0.2, ymax)
    ax.set_xlabel("u")
    ax.set_ylabel(r"$e_q(u)$")
    ax.set_title(r"q-esponenziale  $e_q(u) = [1 + (1-q)\,u]^{1/(1-q)}$")
    ax.grid(True, color="#e0e0e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
    if show:
        plt.show()
    return fig, ax


def plot_q_gaussian(q_values=(0.5, 1.0, 1.5, 2.0), beta=1.0, x=None,
                    logy=False, filename=None, show=True):
    """Traccia e_q(-beta x^2) per ciascun q in q_values (massimo 6 curve)."""
    if len(q_values) > len(COLORS):
        raise ValueError(f"al massimo {len(COLORS)} valori di q")
    if x is None:
        x = np.linspace(-5, 5, 2001) / np.sqrt(beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    for q, color in zip(q_values, COLORS):
        y = q_exp(-beta * x**2, q)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=color, lw=2, label=f"q = {q:g}")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$e_q(-\beta x^2)$")
    ax.set_title(r"$e_q(-\beta x^2) = [1 - (1-q)\,\beta x^2]^{1/(1-q)}$"
                 f"   (β = {beta:g})")
    ax.grid(True, color="#e0e0e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
    if show:
        plt.show()
    return fig, ax


def C_q(q):
    """Costante di normalizzazione C_q = integrale di e_q(-x^2) su R (q < 3)."""
    if q >= 3:
        raise ValueError("C_q è finita solo per q < 3")
    if np.isclose(q, 1.0):
        return np.sqrt(np.pi)
    if q < 1:
        return (2 * np.sqrt(np.pi) * math.gamma(1 / (1 - q))
                / ((3 - q) * np.sqrt(1 - q)
                   * math.gamma((3 - q) / (2 * (1 - q)))))
    return (np.sqrt(np.pi) * math.gamma((3 - q) / (2 * (q - 1)))
            / (np.sqrt(q - 1) * math.gamma(1 / (q - 1))))


def g_q(u, q):
    """q-Gaussiana normalizzata g_q(u) = e_q(-u^2/(3-q)) / (sqrt(3-q) C_q)."""
    u = np.asarray(u, dtype=float)
    return q_exp(-u**2 / (3 - q), q) / (np.sqrt(3 - q) * C_q(q))


def plot_g_q(q_values=(0.5, 1.0, 1.5, 2.0), u=None, logy=False,
             filename=None, show=True):
    """Traccia g_q(u) per ciascun q in q_values (q < 3, massimo 6 curve)."""
    if len(q_values) > len(COLORS):
        raise ValueError(f"al massimo {len(COLORS)} valori di q")
    if u is None:
        u = np.linspace(-6, 6, 2001)

    fig, ax = plt.subplots(figsize=(8, 5))
    for q, color in zip(q_values, COLORS):
        y = g_q(u, q)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(u, y, color=color, lw=2, label=f"q = {q:g}")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel(r"$g_q(u)$")
    ax.set_title(r"$g_q(u) = \frac{1}{\sqrt{3-q}\,C_q}\,"
                 r"e_q\!\left(-\frac{u^2}{3-q}\right)$")
    ax.grid(True, color="#e0e0e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150)
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    plot_q_exponential((0.5, 1.0, 1.5, 2.0), filename="q_exponential.png",
                       show=False)
    plot_q_gaussian((0.5, 1.0, 1.5, 2.0), beta=1.0,
                    filename="q_gaussian.png", show=False)
    plot_q_gaussian((0.5, 1.0, 1.5, 2.0), beta=1.0, logy=True,
                    filename="q_gaussian_log.png", show=False)
    plot_g_q((0.5, 1.0, 1.5, 2.0), filename="g_q.png", show=False)
    plot_g_q((0.5, 1.0, 1.5, 2.0), logy=True, filename="g_q_log.png",
             show=False)
