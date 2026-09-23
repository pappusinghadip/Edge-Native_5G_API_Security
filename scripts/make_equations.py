"""Render the proposal's display equations to PNGs via matplotlib mathtext.

Rendering to images rather than relying on the word processor's equation editor keeps the
notation identical between the .docx and the PDF, which earlier versions did not achieve.
Notation follows Section 2.7 of the round-4 review, except that the ensemble weights are
beta (lambda is the attack scale factor) and TTE_flows = t* - t0, which is what
src/mitigation/agent.py computes and what Chapter 4 reports.
"""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
# STIX matches the Times body text; 11 pt matches the body size
matplotlib.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
import matplotlib.pyplot as plt

OUT = Path("results/figures/eq"); OUT.mkdir(parents=True, exist_ok=True)

# Keys in order of appearance in Chapter 3; numbers are assigned from this order, so a new
# equation must be inserted where it appears in the text.
W = r"\mathbf{w}"
EQ = [
 ("evidence",     r"R_t \;=\; \sum_{i=t-n+1}^{t} \mathbb{1}\!\left[\, p_i \geq \tau \,\right]"),
 ("enforce",      r"M_t \;=\; \left\{\genfrac{}{}{0}{0}{1,\quad R_t \geq k,}{0,\quad R_t < k.}\right."),
 ("posterior",    r"P_\theta(y=1 \mid \mathbf{x}) \;=\; f_\theta(\mathbf{x})"),
 ("softmax",      r"[f_\theta(\mathbf{x})]_c \;=\; \dfrac{\exp(z_c)}{\Sigma_{j=1}^{C}\, \exp(z_j)}"),
 ("focal",        r"\mathcal{L}_{\mathrm{focal}} \;=\; -\,\alpha_t\,(1-p_t)^{\gamma}\,\log(p_t)"),
 ("focalpt",      r"p_t \;=\; y\,p + (1-y)(1-p), \qquad \alpha_t \;=\; y\,\alpha + (1-y)(1-\alpha)"),
 ("decision",     r"\hat{y} \;=\; \mathbb{1}\!\left[\, P_\theta(y=1 \mid \mathbf{x}) \geq \tau \,\right]"),
 ("ensemble",     r"s(\mathbf{x}) \;=\; \sum_{m=1}^{M} \beta_m\, p_m(\mathbf{x}), \qquad \beta_m \geq 0, \qquad \sum_{m=1}^{M} \beta_m = 1"),
 ("weightsearch", r"\beta^{\star} \;=\; \arg\max_{\beta}\; \mathrm{AUC}_{\mathrm{val}}(s_{\beta})"),
 ("complexity",   r"\mathcal{O}\!\left( \sum_{\ell=1}^{L} N_\ell\, K_\ell\, C_{\ell-1}\, C_\ell \right)"),
 ("fedavg",       W + r"^{(t+1)} \;=\; \sum_{k=1}^{K} \dfrac{n_k}{\Sigma_{j=1}^{K}\, n_j}\, " + W + r"_k^{(t+1)}"),
 ("exposure",     r"B_{\mathrm{cen}} \;=\; E\,N, \qquad B_{\mathrm{FL}} \;=\; R\,E_{\mathrm{loc}}\,N"),
 ("attack",       r"\widetilde{\Delta\mathbf{w}}_a^{(t)} \;=\; -\lambda\,\Delta\mathbf{w}_a^{(t)}, \qquad \lambda = 5"),
 ("accepted",     r"\mathcal{A}_t \;=\; \left\{\, k \;:\; \|\Delta" + W + r"_k^{(t)}\|_2 \;\leq\; \kappa\; \mathrm{median}_{j}\, \|\Delta" + W + r"_j^{(t)}\|_2 \,\right\}"),
 ("filtered",     W + r"^{(t+1)} \;=\; \dfrac{\Sigma_{k \in \mathcal{A}_t}\, n_k\, " + W + r"_k^{(t+1)}}{\Sigma_{k \in \mathcal{A}_t}\, n_k}"),
 ("noninf",       r"H_0:\; \mu_{\mathrm{cen}} - \mu_{\mathrm{FL}} \geq \delta \qquad \mathrm{vs.} \qquad H_1:\; \mu_{\mathrm{cen}} - \mu_{\mathrm{FL}} < \delta"),
 ("tstar",        r"t^{\star} \;=\; \min\left\{\, t \geq t_0 \;:\; R_t \geq k \,\right\}"),
 ("ttdflows",     r"\mathrm{TTE}_{\mathrm{flows}} \;=\; t^{\star} - t_0"),
 ("ttdtime",      r"\mathrm{TTE}_{\mathrm{time}} \;=\; T_{t^{\star}} - T_{t_0}"),
 ("ttdrate",      r"\mathrm{TTE}_{\mathrm{time}} \;=\; \dfrac{t^{\star} - t_0}{r}"),
 ("latency",      r"\bar{L} \;=\; \dfrac{1}{N}\sum_{i=1}^{N} \ell_i"),
 ("comm",         r"C_{\mathrm{FL,node}} = 2R\,|" + W + r"|, \qquad C_{\mathrm{central,node}} = D, \qquad \rho = \dfrac{D}{2R\,|" + W + r"|}"),
 ("commtotal",    r"C_{\mathrm{FL,total}} \;=\; 2KR\,|" + W + r"| \;=\; \mathcal{O}(K)"),
]

DPI, CONTENT_IN, PT = 300, 6.5, 11
widths = {}
for i, (key, tex) in enumerate(EQ, start=1):
    num = f"3.{i}"
    fig = plt.figure(figsize=(11.0, 1.2))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.5, rf"${tex}\qquad\qquad\mathrm{{({num})}}$", fontsize=PT,
             ha="center", va="center", color="#1a1a1a")
    p = OUT / f"eq_{key}.png"
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    with open(p, "rb") as fh:
        fh.seek(16); px_w = int.from_bytes(fh.read(4), "big")
    # natural width, so the equation prints at PT points on the page
    widths[key] = round(min(1.0, (px_w / DPI) / CONTENT_IN), 3)
    print(f"wrote {p.name}  ({num})  width_frac={widths[key]}")
(OUT / "eq_widths.json").write_text(json.dumps(widths, indent=1))
