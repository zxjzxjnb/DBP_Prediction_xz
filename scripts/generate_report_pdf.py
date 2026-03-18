"""Generate a clean, vertical (portrait A4) group-meeting PDF report.

Layout philosophy
-----------------
* Portrait A4 (210 × 297 mm) — natural reading direction.
* Single-column throughout: text blocks, then images stacked vertically.
* Every image occupies its own horizontal band — no side-by-side to avoid overlap.
* Auto page-break governs flow; manual _guard() calls prevent orphan headings.
* Modern accent palette: Indigo (#1565C0) headers, slate body, mint highlights.

Covers:
  1. Title page
  2. Study overview & methodology
  3. Ablation performance summary (tables)
  4. SHAP — best model per target (T-THMs, DBCM, BDCM)
  5. Cross-model feature importance (9 features)
  6. Feature importance evolution — Random Forest
  7. Feature importance evolution — XGBoost
  8. Relative share & stacked area
  9. Delta heatmap & R² vs SHAP
  10. Neural network SHAP evolution (MLP, KAN)
  11. Conclusions & recommendations

Usage:
    python scripts/generate_report_pdf.py
"""

from __future__ import annotations

import os
from fpdf import FPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT, "results", "shap_analysis")
OUT_PATH = os.path.join(PROJECT, "results", "DBP_Ablation_SHAP_Report.pdf")

# ---------------------------------------------------------------------------
# Design tokens (mm)
# ---------------------------------------------------------------------------
M = 18          # page margin (left/right/top)
FOOTER_M = 16   # space reserved at page bottom for footer

# Typography
T_PAGE   = 26   # title page main font size
T_SLIDE  = 16   # slide / section heading
T_SUB    = 11   # subtitle / section sub-heading
T_BODY   = 10   # body text
T_SMALL  = 8    # captions, table cells

# Spacing
GAP   = 5       # standard small gap
LGAP  = 10      # larger inter-section gap

# Colors  (R, G, B)
C_ACCENT   = (21, 101, 192)   # deep blue
C_ACCENT2  = (69, 39, 160)    # indigo (alternating)
C_DARK     = (33, 33, 33)     # almost-black body
C_BODY     = (55, 55, 55)
C_MUTED    = (120, 120, 120)
C_TH_BG    = (21, 101, 192)   # table header background
C_TH_FG    = (255, 255, 255)
C_TR_EVEN  = (245, 248, 255)  # table row even
C_HIGHLIGHT= (232, 245, 233)  # highlighted row
C_RULE     = (210, 220, 240)  # horizontal rule
C_CAP      = (100, 110, 130)  # caption text


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------
class ReportPDF(FPDF):
    """Portrait A4, single-column report with clean typographic hierarchy."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(M, M, M)
        self.set_auto_page_break(auto=True, margin=FOOTER_M + 6)
        self._page_title = ""
        self._setup_font()

    # ------------------------------------------------------------------
    # Font setup
    # ------------------------------------------------------------------
    def _setup_font(self):
        import glob
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        ]
        self._uni = None
        for pat in candidates:
            for path in glob.glob(pat):
                try:
                    self.add_font("UF", "",  path, uni=True)
                    self.add_font("UF", "B", path, uni=True)
                    self.add_font("UF", "I", path, uni=True)
                    self._uni = "UF"
                    return
                except Exception:
                    continue

    def sf(self, style="", size=10):
        """Set font conveniently."""
        fam = self._uni if self._uni else "Helvetica"
        self.set_font(fam, style, size)

    def color(self, rgb):
        self.set_text_color(*rgb)

    # ------------------------------------------------------------------
    # Header / footer
    # ------------------------------------------------------------------
    def header(self):
        if self.page == 1:
            return
        self.sf("", 7)
        self.color(C_MUTED)
        self.cell(0, 5, "DBP Prediction — Ablation Study & SHAP Interpretability", align="L")
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.3)
        y = self.get_y() + 5
        self.line(M, y, self.w - M, y)
        self.set_y(y + 3)

    def footer(self):
        if self.page == 1:
            return
        self.set_y(-FOOTER_M)
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.2)
        self.line(M, self.get_y(), self.w - M, self.get_y())
        self.ln(2)
        self.sf("I", 7)
        self.color(C_MUTED)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="C")

    # ------------------------------------------------------------------
    # Content width helpers
    # ------------------------------------------------------------------
    @property
    def cw(self) -> float:
        """Usable content width."""
        return self.w - 2 * M

    def _avail(self) -> float:
        """Remaining vertical space on current page."""
        return self.h - self.get_y() - FOOTER_M - 6

    def _guard(self, need: float):
        """Force new page if not enough space remains."""
        if self._avail() < need:
            self.add_page()

    # ------------------------------------------------------------------
    # Typographic helpers
    # ------------------------------------------------------------------
    def hline(self, color=C_RULE, w=0.3):
        self.set_draw_color(*color)
        self.set_line_width(w)
        self.line(M, self.get_y(), self.w - M, self.get_y())
        self.ln(2)

    def slide_heading(self, number: str, title: str, subtitle: str = ""):
        """Top-of-section heading with accent bar."""
        self._guard(22)
        # Accent left bar
        self.set_fill_color(*C_ACCENT)
        self.rect(M, self.get_y(), 3, 8, "F")
        self.sf("B", T_SLIDE)
        self.color(C_ACCENT)
        self.set_x(M + 5)
        self.cell(0, 8, f"{number}  {title}")
        self.ln(9)
        if subtitle:
            self.sf("I", T_SUB)
            self.color(C_MUTED)
            self.multi_cell(0, 5, subtitle)
            self.ln(2)
        self.hline()
        self.ln(GAP / 2)

    def section_label(self, text: str):
        """Smaller in-page section label."""
        self._guard(10)
        self.sf("B", 10)
        self.color(C_ACCENT2)
        self.cell(0, 6, text)
        self.ln(6)

    def body(self, text: str, indent: int = 0):
        self._guard(8)
        self.sf("", T_BODY)
        self.color(C_BODY)
        if indent:
            self.set_x(M + indent)
        self.multi_cell(self.cw - indent, 5, text)
        self.ln(2)

    def bullet_item(self, bold_part: str, rest: str = "", size: int = T_BODY):
        self._guard(8)
        self.sf("", size)
        self.color(C_BODY)
        self.cell(5, 5, "\u2022")
        self.sf("B", size)
        self.write(5, bold_part)
        if rest:
            self.sf("", size)
            self.write(5, rest)
        self.ln(5)

    def caption(self, text: str, center=True):
        self.sf("I", T_SMALL)
        self.color(C_CAP)
        self.multi_cell(0, 4, text, align="C" if center else "L")
        self.ln(3)

    def vgap(self, h: float = GAP):
        self.ln(h)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------
    def table_header(self, cols, widths, height=6):
        self._guard(height + 5)
        self.set_fill_color(*C_TH_BG)
        self.set_text_color(*C_TH_FG)
        self.sf("B", T_SMALL)
        for col, w in zip(cols, widths):
            self.cell(w, height, col, border=1, align="C", fill=True)
        self.ln()

    def table_row(self, cells, widths, highlight=False, bold_first=False, row_idx=0):
        bg = C_HIGHLIGHT if highlight else (C_TR_EVEN if row_idx % 2 == 0 else (255, 255, 255))
        self.set_fill_color(*bg)
        self.set_text_color(*C_DARK)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.sf("B" if bold_first and i == 0 else "", T_SMALL)
            self.cell(w, 5, str(cell)[:30], border=1, align="C", fill=True)
        self.ln()

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def _fit(self, path: str, max_w: float, max_h: float):
        """Return (w, h) preserving aspect ratio within max_w × max_h."""
        try:
            im = Image.open(path)
            iw, ih = im.size
            if ih == 0:
                return max_w, max_h * 0.5
            ar = ih / iw
            w = min(max_w, max_h / ar)
            h = w * ar
            if h > max_h:
                h = max_h
                w = h / ar
            return w, h
        except Exception:
            return max_w, max_h * 0.5

    def img_full(self, path: str, caption: str = "", max_h: float = 90):
        """Single centered image spanning full content width."""
        if not os.path.exists(path):
            self.sf("I", 9); self.color((200, 60, 60))
            self.cell(0, 6, f"[Image not found: {os.path.basename(path)}]")
            self.ln(6)
            return
        w, h = self._fit(path, self.cw, max_h)
        self._guard(h + (10 if caption else 4))
        x0 = M + (self.cw - w) / 2
        y0 = self.get_y()
        # Light grey border frame
        self.set_draw_color(220, 225, 235)
        self.set_line_width(0.2)
        self.rect(x0 - 1, y0 - 1, w + 2, h + 2)
        self.image(path, x=x0, y=y0, w=w, h=h)
        self.set_y(y0 + h + 3)
        if caption:
            self.caption(caption)
        else:
            self.ln(GAP)

    def img_pair(self, path1: str, path2: str, cap1: str = "", cap2: str = "",
                 max_h: float = 72):
        """Two images side by side, equal width, NO overlap guaranteed."""
        half_w = (self.cw - GAP) / 2
        _, h1 = self._fit(path1, half_w, max_h) if os.path.exists(path1) else (half_w, max_h)
        _, h2 = self._fit(path2, half_w, max_h) if os.path.exists(path2) else (half_w, max_h)
        row_h = max(h1, h2)
        cap_h = 10 if (cap1 or cap2) else 0
        self._guard(row_h + cap_h + 6)
        y0 = self.get_y()

        for path, x_off, cap in [
            (path1, 0, cap1),
            (path2, half_w + GAP, cap2),
        ]:
            if os.path.exists(path):
                w, h = self._fit(path, half_w, max_h)
                x = M + x_off + (half_w - w) / 2  # center within half
                self.set_draw_color(220, 225, 235)
                self.set_line_width(0.2)
                self.rect(x - 1, y0 - 1, w + 2, h + 2)
                self.image(path, x=x, y=y0, w=w, h=h)
            else:
                self.set_y(y0)
                self.sf("I", 8); self.color((200, 60, 60))
                self.cell(half_w, 5, f"[Missing: {os.path.basename(path)}]")

        # Advance cursor below both images, then add captions side by side
        self.set_y(y0 + row_h + 3)
        if cap1 or cap2:
            self.sf("I", T_SMALL)
            self.color(C_CAP)
            self.cell(half_w, 4, cap1, align="C")
            self.cell(GAP)
            self.cell(half_w, 4, cap2, align="C")
            self.ln(8)
        else:
            self.ln(GAP)


# ---------------------------------------------------------------------------
# Shorthand path helper
# ---------------------------------------------------------------------------
def sp(rel: str) -> str:
    return os.path.join(SHAP_DIR, rel)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — TITLE
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Decorative top bar
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(0, 0, pdf.w, 8, "F")

    pdf.ln(50)

    pdf.sf("B", T_PAGE)
    pdf.color(C_ACCENT)
    pdf.cell(0, 14, "DBP Prediction", align="C"); pdf.ln()
    pdf.cell(0, 14, "Ablation Study & SHAP Analysis", align="C"); pdf.ln(10)

    pdf.hline(C_ACCENT, 0.5)
    pdf.ln(6)

    pdf.sf("", 13)
    pdf.color(C_BODY)
    pdf.cell(0, 7, "Incremental Feature Selection for T-THMs, DBCM, and BDCM", align="C"); pdf.ln(7)
    pdf.sf("", 11)
    pdf.color(C_MUTED)
    pdf.cell(0, 6, "Models: Random Forest  |  XGBoost  |  MLP  |  KAN", align="C"); pdf.ln(12)

    pdf.sf("I", 10)
    pdf.color(C_MUTED)
    pdf.cell(0, 6, "Group Meeting Report  —  March 2026", align="C")

    # Decorative bottom bar
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(0, pdf.h - 8, pdf.w, 8, "F")

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 2 — STUDY OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("1", "Study Overview", "Motivation, dataset & experimental methodology")

    pdf.section_label("Objective")
    pdf.body(
        "Systematically evaluate the contribution of each water quality parameter to DBP "
        "(Disinfection By-Product) prediction by incrementally adding features and tracking "
        "both model performance (R², RMSE) and feature importance (SHAP values) across "
        "four model families."
    )

    pdf.vgap(GAP)
    pdf.section_label("Dataset")
    cols = ["Property", "Value"]
    ws = [50, pdf.cw - 50]
    pdf.table_header(cols, ws)
    for i, (k, v) in enumerate([
        ("Source",        "DWTP-B (single water treatment plant)"),
        ("Total samples", "175   (141 train  /  34 test — predefined split)"),
        ("Targets",       "T-THMs • DBCM (chlorinated) • BDCM (brominated)"),
        ("All 9 features","pH, UV254, Temperature, TOC, COD, Br⁻, NH₄-N, NO₂-N, NO₃-N"),
        ("Evaluation",    "Test R²,  Test RMSE,  5-fold CV for tuning"),
    ]):
        pdf.table_row([k, v], ws, row_idx=i)
    pdf.vgap(LGAP)

    pdf.section_label("Ablation Rounds — Incremental Feature Addition")
    cols2 = ["Round", "Feature Set", "Added Feature", "Rationale"]
    ws2 = [18, 75, 32, pdf.cw - 125]
    pdf.table_header(cols2, ws2)
    for i, row in enumerate([
        ["R1 (3-feat)", "pH + UV254 + Temperature",              "—",         "Core water-quality indicators"],
        ["R2 (4-feat)", "+ TOC",                                  "+ TOC",     "Dissolved organic carbon"],
        ["R3 (5-feat)", "+ COD",                                  "+ COD",     "Organic demand / reactivity"],
        ["R4 (6-feat)", "+ Br⁻",                                  "+ Br⁻",     "Bromine precursor for BDCM"],
        ["R5 (9-feat)", "+ NH₄-N + NO₂-N + NO₃-N",               "+ Nitrogen","Chloramine formation pathway"],
    ]):
        pdf.table_row(row, ws2, row_idx=i)
    pdf.vgap(LGAP)

    pdf.section_label("Methodology")
    for bold, rest in [
        ("Per-round pipeline: ",
         "Scout run (3-fold CV, ~30 trials) -> Formal tuning (5-fold CV, 50-80 trials)"),
        ("Tuner: ",
         "Optuna TPE sampler with median pruning; stability penalty λ = 0.18"),
        ("Strategy: ",
         "Per-target — independent model for each of the 3 DBP targets"),
        ("Interpretability: ",
         "SHAP (TreeExplainer for RF/XGB; KernelExplainer for MLP/KAN)"),
    ]:
        pdf.bullet_item(bold, rest)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 3 — PERFORMANCE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("2", "Ablation Performance Summary",
                      "Best-model test R² and RMSE per target across feature rounds")

    for target_label, rows, best_idx in [
        (
            "Target: T-THMs (Total Trihalomethanes)",
            [
                ["3-feat", "RF",  "0.545", "6.90", "—"],
                ["4-feat (+TOC)", "RF",  "0.549", "6.86", "+0.004"],
                ["5-feat (+COD)", "RF",  "0.571", "6.70", "+0.022 (*)"],
                ["6-feat (+Br⁻)", "RF",  "0.492", "7.29", "−0.079"],
                ["9-feat (+N)",   "XGB", "0.533", "6.98", "+0.041"],
            ], 2,
        ),
        (
            "Target: DBCM (Dibromochloromethane)",
            [
                ["3-feat", "MLP", "0.597", "2.46", "—"],
                ["4-feat (+TOC)", "MLP", "0.638", "2.33", "+0.041"],
                ["5-feat (+COD)", "MLP", "0.659", "2.27", "+0.021 (*)"],
                ["6-feat (+Br⁻)", "KAN", "0.566", "2.56", "−0.093"],
                ["9-feat (+N)",   "XGB", "0.561", "2.57", "−0.005"],
            ], 2,
        ),
        (
            "Target: BDCM (Bromodichloromethane)",
            [
                ["3-feat", "XGB", "0.314", "1.72", "—"],
                ["4-feat (+TOC)", "RF",  "0.386", "1.62", "+0.072"],
                ["5-feat (+COD)", "RF",  "0.393", "1.61", "+0.007"],
                ["6-feat (+Br⁻)", "MLP", "0.434", "1.56", "+0.041 (*)"],
                ["9-feat (+N)",   "RF",  "0.391", "1.62", "−0.043"],
            ], 3,
        ),
    ]:
        pdf._guard(14 + len(rows) * 5 + 6)
        pdf.section_label(target_label)
        hcols = ["Feature Set", "Best Model", "R²", "RMSE", "ΔR² vs Prev"]
        hws = [45, 28, 18, 18, pdf.cw - 109]
        pdf.table_header(hcols, hws)
        for i, row in enumerate(rows):
            pdf.table_row(row, hws, highlight=(i == best_idx), bold_first=True, row_idx=i)
        pdf.vgap(LGAP)

    pdf.section_label("Key Takeaway")
    for bold, rest in [
        ("Optimal feature count: ",
         "5 features for T-THMs and DBCM; 6 features (+ Br⁻) for BDCM."),
        ("Nitrogen indicators: ",
         "NH₄-N, NO₂-N, NO₃-N add no performance gain on this dataset."),
        ("6-feature regression: ",
         "Adding Br⁻ hurts T-THMs/DBCM — Bromine acts as noise for chlorinated DBPs."),
    ]:
        pdf.bullet_item(bold, rest)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 4 — SHAP BEST PER TARGET: T-THMs
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("3", "SHAP — Best Model per Target",
                      "T-THMs: Random Forest with 5 features")

    pdf.section_label("Feature Importance Bar Chart")
    pdf.img_full(sp("best_per_target/T_THMs_ug_L/bar_importance.png"),
                 "Fig 1a  Mean |SHAP| values — T-THMs (RF, 5-feat)", max_h=85)

    pdf.section_label("SHAP Beeswarm Summary")
    pdf.img_full(sp("best_per_target/T_THMs_ug_L/summary_beeswarm.png"),
                 "Fig 1b  SHAP beeswarm: feature value vs impact on prediction — T-THMs", max_h=85)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 5 — SHAP BEST PER TARGET: DBCM
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("3", "SHAP — Best Model per Target (cont.)",
                      "DBCM: MLP with 5 features")

    pdf.section_label("Feature Importance Bar Chart")
    pdf.img_full(sp("best_per_target/DBCM_ug_L/bar_importance.png"),
                 "Fig 2a  Mean |SHAP| values — DBCM (MLP, 5-feat)", max_h=85)

    pdf.section_label("SHAP Beeswarm Summary")
    pdf.img_full(sp("best_per_target/DBCM_ug_L/summary_beeswarm.png"),
                 "Fig 2b  SHAP beeswarm — DBCM (MLP, 5-feat)", max_h=85)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 6 — SHAP BEST PER TARGET: BDCM
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("3", "SHAP — Best Model per Target (cont.)",
                      "BDCM: MLP with 6 features (includes Br⁻)")

    pdf.section_label("Feature Importance Bar Chart")
    pdf.img_full(sp("best_per_target/BDCM_ug_L/bar_importance.png"),
                 "Fig 3a  Mean |SHAP| values — BDCM (MLP, 6-feat)", max_h=85)

    pdf.section_label("SHAP Beeswarm Summary")
    pdf.img_full(sp("best_per_target/BDCM_ug_L/summary_beeswarm.png"),
                 "Fig 3b  SHAP beeswarm — BDCM (MLP, 6-feat)", max_h=85)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 7 — SHAP DEPENDENCE PLOTS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("3", "SHAP Dependence Plots",
                      "Key feature-target non-linear interactions")

    for dep_path, fig_label, cap in [
        (sp("best_per_target/T_THMs_ug_L/dep_top1_temp_C.png"),
         "T-THMs  ×  Temperature",
         "Fig 4a  Temperature vs SHAP impact on T-THMs (RF, 5-feat)"),
        (sp("best_per_target/DBCM_ug_L/dep_top1_temp_C.png"),
         "DBCM  ×  Temperature",
         "Fig 4b  Temperature vs SHAP impact on DBCM (MLP, 5-feat)"),
        (sp("best_per_target/BDCM_ug_L/dep_top1_temp_C.png"),
         "BDCM  ×  Temperature",
         "Fig 4c  Temperature vs SHAP impact on BDCM (MLP, 6-feat)"),
    ]:
        pdf.section_label(fig_label)
        pdf.img_full(dep_path, cap, max_h=70)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 8 — CROSS-MODEL AT 9 FEATURES
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("4", "Cross-Model Feature Importance (9 Features)",
                      "Consensus across RF, XGBoost, MLP, and KAN")

    for fig_id, target, path in [
        ("5a", "T-THMs",  sp("9feat/cross_model_T_THMs_ug_L.png")),
        ("5b", "DBCM",    sp("9feat/cross_model_DBCM_ug_L.png")),
        ("5c", "BDCM",    sp("9feat/cross_model_BDCM_ug_L.png")),
    ]:
        pdf.section_label(f"Target: {target}")
        pdf.img_full(path, f"Fig {fig_id}  All-model SHAP importance — {target} (9-feat)", max_h=72)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 9 — EVOLUTION: RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("5", "Feature Importance Evolution — Random Forest",
                      "Mean |SHAP| per feature as ablation round increases from 3 to 9")

    for fig_id, target, path in [
        ("6a", "T-THMs",  sp("cross_ablation/rf/evolution_T_THMs_ug_L.png")),
        ("6b", "DBCM",    sp("cross_ablation/rf/evolution_DBCM_ug_L.png")),
        ("6c", "BDCM",    sp("cross_ablation/rf/evolution_BDCM_ug_L.png")),
    ]:
        pdf.section_label(f"Target: {target}")
        pdf.img_full(path, f"Fig {fig_id}  RF importance evolution — {target}", max_h=72)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 10 — EVOLUTION: XGBOOST
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("5", "Feature Importance Evolution — XGBoost",
                      "Consistent patterns with Random Forest; confirms cross-model consensus")

    for fig_id, target, path in [
        ("7a", "T-THMs",  sp("cross_ablation/xgb/evolution_T_THMs_ug_L.png")),
        ("7b", "DBCM",    sp("cross_ablation/xgb/evolution_DBCM_ug_L.png")),
        ("7c", "BDCM",    sp("cross_ablation/xgb/evolution_BDCM_ug_L.png")),
    ]:
        pdf.section_label(f"Target: {target}")
        pdf.img_full(path, f"Fig {fig_id}  XGBoost importance evolution — {target}", max_h=72)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 11 — RELATIVE SHARE
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("6", "Relative Feature Share (100% Stacked)",
                      "How core feature importance redistributes as new features are added")

    pdf.section_label("Random Forest — Relative Importance Shift")
    pdf.img_pair(
        sp("cross_ablation/rf/relative_T_THMs_ug_L.png"),
        sp("cross_ablation/rf/relative_DBCM_ug_L.png"),
        "T-THMs (RF)", "DBCM (RF)", max_h=75,
    )
    pdf.img_full(sp("cross_ablation/rf/relative_BDCM_ug_L.png"),
                 "Fig 8c  Relative importance — BDCM (RF)", max_h=75)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 12 — STACKED AREA
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("6", "Stacked Area — Total Feature Importance",
                      "Absolute SHAP stacked area across ablation rounds")

    pdf.section_label("Random Forest")
    pdf.img_pair(
        sp("cross_ablation/rf/stacked_T_THMs_ug_L.png"),
        sp("cross_ablation/rf/stacked_DBCM_ug_L.png"),
        "T-THMs (RF)", "DBCM (RF)", max_h=75,
    )
    pdf.img_full(sp("cross_ablation/rf/stacked_BDCM_ug_L.png"),
                 "Fig 9c  Stacked area — BDCM (RF)", max_h=72)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 13 — DELTA HEATMAP
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("7", "Core Feature Displacement — Delta Heatmap",
                      "Change in |SHAP| of pH, UV254, Temperature when a new feature is introduced (RF)")

    pdf.img_full(sp("cross_ablation/rf/delta_heatmap.png"),
                 "Fig 10  Delta heatmap: ΔMean|SHAP| of core features after each addition (RF)", max_h=100)

    pdf.section_label("Observations")
    for bold, rest in [
        ("Temperature is robust: ",
         "its SHAP stays relatively stable across rounds — not easily displaced."),
        ("UV254 dilution: ",
         "loses importance once TOC is introduced (TOC and UV254 are correlated proxies)."),
        ("pH: ",
         "shows minor negative displacement but remains secondary across all rounds."),
    ]:
        pdf.bullet_item(bold, rest)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 14 — R² vs SHAP
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("7", "Model Performance vs SHAP Complexity",
                      "Test R² plotted against total |SHAP| — diminishing returns beyond 5–6 features")

    pdf.section_label("Random Forest")
    pdf.img_full(sp("cross_ablation/rf/r2_vs_shap.png"),
                 "Fig 11a  RF: R² vs total |SHAP|  —  peak at 5 features, inflation at 9", max_h=90)

    pdf.section_label("XGBoost")
    pdf.img_full(sp("cross_ablation/xgb/r2_vs_shap.png"),
                 "Fig 11b  XGBoost: same diminishing-return pattern", max_h=90)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 15 — NEURAL NETWORK SHAP EVOLUTION
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("8", "Neural Network SHAP Evolution",
                      "MLP and KAN — importance patterns across ablation rounds")

    pdf.section_label("MLP — T-THMs")
    pdf.img_full(sp("cross_ablation/mlp/evolution_T_THMs_ug_L.png"),
                 "Fig 12a  MLP feature importance evolution — T-THMs", max_h=75)

    pdf.section_label("KAN — BDCM")
    pdf.img_full(sp("cross_ablation/kan/evolution_BDCM_ug_L.png"),
                 "Fig 12b  KAN feature importance evolution — BDCM", max_h=75)

    pdf.section_label("Observations")
    for bold, rest in [
        ("Temperature: ",
         "dominant in both MLP and KAN; sharper concentration than tree models."),
        ("KAN <-> NO₃-N: ",
         "KAN assigns notable importance to NO₃-N for BDCM at 9-feat (unique to KAN)."),
        ("Consensus: ",
         "all 4 architectures agree — Temperature >> TOC/COD >> remaining features."),
    ]:
        pdf.bullet_item(bold, rest)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 16 — BEESWARM GRIDS (RF)
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("8", "SHAP Beeswarm Grid — Random Forest",
                      "Feature distributions per ablation round, all three targets")

    for fig_id, target, path in [
        ("13a", "T-THMs", sp("cross_ablation/rf/beeswarm_grid_T_THMs_ug_L.png")),
        ("13b", "DBCM",   sp("cross_ablation/rf/beeswarm_grid_DBCM_ug_L.png")),
        ("13c", "BDCM",   sp("cross_ablation/rf/beeswarm_grid_BDCM_ug_L.png")),
    ]:
        pdf.section_label(f"Target: {target}")
        pdf.img_full(path, f"Fig {fig_id}  RF beeswarm grid — {target}", max_h=80)

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 17 — CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.slide_heading("9", "Conclusions & Recommendations",
                      "Summary of findings and proposed next steps")

    pdf.section_label("Optimal Feature Sets per Target")
    cols3 = ["Target", "Best Feature Set", "Model", "R²", "Key Driver"]
    ws3 = [20, 68, 18, 14, pdf.cw - 120]
    pdf.table_header(cols3, ws3)
    for i, row in enumerate([
        ["T-THMs", "pH + UV254 + Temp + TOC + COD",          "RF",  "0.571", "Temperature"],
        ["DBCM",   "pH + UV254 + Temp + TOC + COD",          "MLP", "0.659", "Temp + TOC"],
        ["BDCM",   "pH + UV254 + Temp + TOC + COD + Br⁻",    "MLP", "0.434", "Temp + Br⁻"],
    ]):
        pdf.table_row(row, ws3, highlight=True, row_idx=i)
    pdf.vgap(LGAP)

    pdf.section_label("Key Findings")
    for bold, rest in [
        ("1. Temperature is dominant: ",
         "accounts for 55–80% of total SHAP across all models and targets."),
        ("2. 5-feature optimum: ",
         "Peak performance for T-THMs and DBCM — extra features add noise not signal."),
        ("3. Br⁻ is target-specific: ",
         "Essential for BDCM; harmful (dilution effect) for chlorinated DBPs."),
        ("4. Nitrogen irrelevant: ",
         "NH₄-N, NO₂-N, NO₃-N deliver no measurable gain on DWTP-B."),
        ("5. Cross-model consensus: ",
         "All 4 architectures agree on the top feature ranking."),
        ("6. BDCM bottleneck: ",
         "All models plateau at R² ≈ 0.43; may indicate data limitations or noise."),
    ]:
        pdf.bullet_item(bold, rest)
    pdf.vgap(GAP)

    pdf.section_label("Recommended Next Steps")
    for text in [
        "SHAP interaction values: quantify Temperature × TOC and Temperature × Br⁻ synergies.",
        "Cross-plant validation: test the 5-feature model on DWTP-A data.",
        "Feature engineering: Temp × UV254 interaction term, log(TOC), Br⁻/TOC ratio.",
        "Collect more BDCM samples to improve model resolution for brominated DBPs.",
        "Ensemble: stack RF + XGBoost predictions as a simple performance upper bound.",
    ]:
        pdf.bullet_item("• ", text)

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════
    pdf.output(OUT_PATH)
    print(f"\n✅  Report saved → {OUT_PATH}")


if __name__ == "__main__":
    build_report()
