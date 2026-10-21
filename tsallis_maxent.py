"""Distribuzione di massima entropia di Tsallis (q-Gaussiana).

Massimizza  S_q[f] = (1 - ∫ f(x)^q dx) / (q - 1)
con i vincoli
    ∫ f(x) dx = 1                                  (normalizzazione)
    ∫ x f_q(x) dx = mu_q                           (q-media)
    ∫ (x - mu_q)^2 f_q(x) dx = sigma_q^2           (q-varianza)
dove f_q = f^q / ∫ f^q è la distribuzione escort.

Soluzione:
    f(x) = [1 - (1-q) beta (x-mu_q)^2]_+^{1/(1-q)} / Z_q,
    beta = 1 / ((3-q) sigma_q^2),   valida per q < 3.
"""
import math
import numpy as np


def q_exp(x, q):
    """q-esponenziale e_q(x) = [1 + (1-q) x]_+^{1/(1-q)}."""
    x = np.asarray(x, dtype=float)
    if np.isclose(q, 1.0):
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    out = np.zeros_like(base)
    pos = base > 0
    out[pos] = base[pos] ** (1.0 / (1.0 - q))
    return out


def _norm_const(q, beta):
    """Z_q = ∫ e_q(-beta x^2) dx."""
    if np.isclose(q, 1.0):
        return math.sqrt(math.pi / beta)
    if q < 1:
        logc = (math.lgamma(1 / (1 - q)) - math.lgamma((3 - q) / (2 * (1 - q))))
        c = 2 * math.sqrt(math.pi) * math.exp(logc) / ((3 - q) * math.sqrt(1 - q))
    else:  # 1 < q < 3
        logc = (math.lgamma((3 - q) / (2 * (q - 1))) - math.lgamma(1 / (q - 1)))
        c = math.sqrt(math.pi) * math.exp(logc) / math.sqrt(q - 1)
    return c / math.sqrt(beta)


def tsallis_maxent_pdf(x, q, mu_q=0.0, sigma_q=1.0):
    """Densità f(x) che massimizza l'entropia di Tsallis S_q
    dati normalizzazione, q-media mu_q e q-deviazione standard sigma_q."""
    if q >= 3:
        raise ValueError("q deve essere < 3 (q-varianza finita)")
    if sigma_q <= 0:
        raise ValueError("sigma_q deve essere > 0")
    beta = 1.0 / ((3.0 - q) * sigma_q ** 2)
    x = np.asarray(x, dtype=float)
    return q_exp(-beta * (x - mu_q) ** 2, q) / _norm_const(q, beta)


def tsallis_maxent_pdf_asym(x, q, sigma_minus=1.0, sigma_plus=1.0, eta=0.0):
    """Densità f(x) che massimizza l'entropia di Tsallis S_q dati
    normalizzazione e q-semideviazioni sinistra/destra attorno a eta:
        sigma_minus^2 = ∫_{x<eta} (x-eta)^2 f^q dx / ∫_{x<eta} f^q dx
        sigma_plus^2  = ∫_{x>eta} (x-eta)^2 f^q dx / ∫_{x>eta} f^q dx
    Soluzione: q-Gaussiana a due pezzi, continua in eta,
        f(x) = e_q(-beta_pm (x-eta)^2) / Z,  beta_pm = 1/((3-q) sigma_pm^2),
        Z = (Z_q(beta_-) + Z_q(beta_+)) / 2."""
    if q >= 3:
        raise ValueError("q deve essere < 3 (q-varianza finita)")
    if sigma_minus <= 0 or sigma_plus <= 0:
        raise ValueError("sigma_minus e sigma_plus devono essere > 0")
    beta_m = 1.0 / ((3.0 - q) * sigma_minus ** 2)
    beta_p = 1.0 / ((3.0 - q) * sigma_plus ** 2)
    Z = 0.5 * (_norm_const(q, beta_m) + _norm_const(q, beta_p))
    x = np.asarray(x, dtype=float)
    beta = np.where(x < eta, beta_m, beta_p)
    return q_exp(-beta * (x - eta) ** 2, q) / Z


if __name__ == "__main__":
    from scipy.integrate import quad

    mu, s = 1.5, 0.8
    for q in (0.5, 1.0, 1.5, 2.0, 2.5):
        f = lambda x: float(tsallis_maxent_pdf(x, q, mu, s))
        I = lambda g: quad(g, -np.inf, np.inf, limit=500, points=None)[0] if q >= 1 else \
            quad(g, mu - 50 * s, mu + 50 * s, limit=500, points=[mu])[0]
        norm = I(f)
        cq = I(lambda x: f(x) ** q)
        m = I(lambda x: x * f(x) ** q) / cq
        v = I(lambda x: (x - mu) ** 2 * f(x) ** q) / cq
        print(f"q={q}: ∫f={norm:.6f}  mu_q={m:.6f}  sigma_q={math.sqrt(v):.6f}")
