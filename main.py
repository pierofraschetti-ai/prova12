"""Script principale: calcola e disegna la q-Gaussiana per diversi q."""
import numpy as np

from tsallis_maxent import tsallis_maxent_pdf, tsallis_maxent_pdf_asym
from plot_tsallis import plot_tsallis_pdfs, plot_tsallis_asym_pdfs


def main():
    mu_q, sigma_q = 0.0, 1.0
    q_values = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Valori della densità in alcuni punti
    x = np.array([0.0, 1.0, 3.0])
    for q in q_values:
        print(f"q = {q}: f({x}) = {tsallis_maxent_pdf(x, q, mu_q, sigma_q)}")

    show = __import__("matplotlib").get_backend().lower() != "agg"
    plot_tsallis_pdfs(q_values, mu_q, sigma_q,
                      filename="tsallis_pdf.png", show=show)
    plot_tsallis_pdfs(q_values, mu_q, sigma_q, logy=True,
                      filename="tsallis_pdf_log.png", show=show)

    # Caso asimmetrico: q-semideviazioni sinistra/destra attorno a eta
    eta, sigma_minus, sigma_plus = 0.0, 4 / 5, 6 / 5
    for q in q_values:
        f = tsallis_maxent_pdf_asym(x, q, sigma_minus, sigma_plus, eta)
        print(f"[asim] q = {q}: f({x}) = {f}")

    plot_tsallis_asym_pdfs(q_values, sigma_minus, sigma_plus, eta,
                           filename="tsallis_pdf_asym.png", show=show)
    plot_tsallis_asym_pdfs(q_values, sigma_minus, sigma_plus, eta, logy=True,
                           filename="tsallis_pdf_asym_log.png", show=show)


if __name__ == "__main__":
    main()
