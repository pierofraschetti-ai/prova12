"""Grafico della q-Gaussiana di massima entropia di Tsallis per diversi q."""
import numpy as np
import matplotlib.pyplot as plt

from tsallis_maxent import tsallis_maxent_pdf, tsallis_maxent_pdf_asym

# Palette categorica in ordine fisso (non ciclata)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300"]


def plot_tsallis_pdfs(q_values, mu_q=0.0, sigma_q=1.0, x=None,
                      logy=False, filename=None, show=True):
    """Traccia f(x) per ciascun q in q_values (massimo 6 curve)."""
    if len(q_values) > len(COLORS):
        raise ValueError(f"al massimo {len(COLORS)} valori di q")
    if x is None:
        x = np.linspace(mu_q - 6 * sigma_q, mu_q + 6 * sigma_q, 2001)

    fig, ax = plt.subplots(figsize=(8, 5))
    for q, color in zip(q_values, COLORS):
        y = tsallis_maxent_pdf(x, q, mu_q, sigma_q)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=color, lw=2, label=f"q = {q:g}")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Massima entropia di Tsallis  (μ_q = {mu_q:g}, σ_q = {sigma_q:g})")
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


def plot_tsallis_asym_pdfs(q_values, sigma_minus=0.8, sigma_plus=1.2, eta=0.0,
                           x=None, logy=False, filename=None, show=True):
    """Traccia la q-Gaussiana asimmetrica f(x) per ciascun q in q_values
    (massimo 6 curve)."""
    if len(q_values) > len(COLORS):
        raise ValueError(f"al massimo {len(COLORS)} valori di q")
    if x is None:
        x = np.linspace(eta - 6 * sigma_minus, eta + 6 * sigma_plus, 2001)

    fig, ax = plt.subplots(figsize=(8, 5))
    for q, color in zip(q_values, COLORS):
        y = tsallis_maxent_pdf_asym(x, q, sigma_minus, sigma_plus, eta)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=color, lw=2, label=f"q = {q:g}")

    ax.axvline(eta, color="#9e9e9e", lw=1, ls="--")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Massima entropia di Tsallis asimmetrica  "
                 f"(η = {eta:g}, σ_q− = {sigma_minus:g}, σ_q+ = {sigma_plus:g})")
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
