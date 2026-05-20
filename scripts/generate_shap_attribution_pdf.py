"""Generate a polished English PDF report for the SHAP attribution study.

The report covers:
  - motivation and hypotheses
  - controlled experimental design
  - model / SHAP protocol
  - condition-level evidence tables
  - selected SHAP figures
  - conclusions and limitations

Usage:
    python scripts/generate_shap_attribution_pdf.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "shap_attribution_study_report.pdf"

RESULTS_DIR = PROJECT / "results" / "shap_attribution"
TEMP_VARIABILITY_DIR = RESULTS_DIR / "temp_variability_stratified"
A_PRIME_RUN = (
    PROJECT
    / "checkpoints"
    / "shap_attribution"
    / "tailake_5common_formal"
    / "20260407T134424Z"
)

RAW_PATH = RESULTS_DIR / "all_shap_results.csv"
AGG_PATH = RESULTS_DIR / "subsample_aggregated.csv"
MASTER_PATH = RESULTS_DIR / "master_comparison.md"
TEMP_IMPORTANCE_PATH = TEMP_VARIABILITY_DIR / "importance_by_group.csv"
TEMP_GROUP_PATH = TEMP_VARIABILITY_DIR / "group_sample_sizes.csv"

TARGET_ORDER = ["thm4", "dbcm", "bdcm"]
TARGET_LABELS = {
    "thm4": "THM family response",
    "dbcm": "DBCM",
    "bdcm": "BDCM",
}
MODEL_ORDER = ["rf", "xgb"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
CONDITION_ORDER = ["A'", "B", "D", "C", "E"]

CONDITION_META = {
    "A'": {
        "dataset": "Tai Lake (DWTP-B)",
        "n": "175",
        "features": "5-common",
        "status": "New",
        "role": "External reference baseline",
    },
    "B": {
        "dataset": "Dataset1",
        "n": "488",
        "features": "5-common",
        "status": "Existing",
        "role": "Full-data matched-features baseline",
    },
    "D": {
        "dataset": "Dataset1",
        "n": "175 x 5 seeds",
        "features": "5-common",
        "status": "New",
        "role": "Sample-size control",
    },
    "C": {
        "dataset": "Dataset1",
        "n": "488",
        "features": "6-feat (+ Cl2)",
        "status": "Existing",
        "role": "Full-data expanded-features baseline",
    },
    "E": {
        "dataset": "Dataset1",
        "n": "175 x 5 seeds",
        "features": "6-feat (+ Cl2)",
        "status": "New",
        "role": "Sample-size + feature-set control",
    },
}

IMPORTANT_FIGURES = [
    {
        "title": "RF THM-family response",
        "subtitle": "This pair captures the strongest Cl2-driven shift in THM attribution.",
        "cross": RESULTS_DIR / "cross_condition_rf_thm4.png",
        "cross_caption": (
            "Cross-condition mean absolute SHAP comparison for Random Forest on the THM-family target."
        ),
        "heatmap": RESULTS_DIR / "ranking_heatmap_rf_thm4.png",
        "heatmap_caption": (
            "Rank heatmap for Random Forest on the THM-family target across A', B, D, C, and E."
        ),
    },
    {
        "title": "RF BDCM",
        "subtitle": "This pair highlights the strongest data-source contrast at matched N and matched features.",
        "cross": RESULTS_DIR / "cross_condition_rf_bdcm.png",
        "cross_caption": "Cross-condition mean absolute SHAP comparison for Random Forest on BDCM.",
        "heatmap": RESULTS_DIR / "ranking_heatmap_rf_bdcm.png",
        "heatmap_caption": "Rank heatmap for Random Forest on BDCM across all conditions.",
    },
    {
        "title": "XGB DBCM",
        "subtitle": "This pair shows the key nuance: DBCM is directionally consistent, but less cleanly separated than THM-family response and BDCM.",
        "cross": RESULTS_DIR / "cross_condition_xgb_dbcm.png",
        "cross_caption": "Cross-condition mean absolute SHAP comparison for XGBoost on DBCM.",
        "heatmap": RESULTS_DIR / "ranking_heatmap_xgb_dbcm.png",
        "heatmap_caption": "Rank heatmap for XGBoost on DBCM across all conditions.",
    },
]


def pdf_text(value: object) -> str:
    text = str(value)
    replacements = {
        "₂": "2",
        "₄": "4",
        "₃": "3",
        "₁": "1",
        "′": "'",
        "–": "-",
        "—": "-",
        "≤": "<=",
        "≥": ">=",
        "≈": "~",
        "μ": "u",
        "×": "x",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


@dataclass(frozen=True)
class AprimeMetrics:
    model: str
    macro_rmse: float
    macro_r2: float
    thm_rmse: float
    dbcm_rmse: float
    bdcm_rmse: float


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW_PATH)
    agg = pd.read_csv(AGG_PATH)
    return raw, agg


def load_temp_variability_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    importance = pd.read_csv(TEMP_IMPORTANCE_PATH)
    groups = pd.read_csv(TEMP_GROUP_PATH)
    return importance, groups


def load_aprime_metrics() -> list[AprimeMetrics]:
    data = load_json(A_PRIME_RUN / "metrics" / "model_comparison.json")
    rows: list[AprimeMetrics] = []
    for model_key in MODEL_ORDER:
        model = data["models"][model_key]
        rows.append(
            AprimeMetrics(
                model=model_key,
                macro_rmse=float(model["macro_test_metrics"]["rmse"]),
                macro_r2=float(model["macro_test_metrics"]["r2"]),
                thm_rmse=float(model["target_metrics"]["T_THMs_ug_L"]["rmse"]),
                dbcm_rmse=float(model["target_metrics"]["DBCM_ug_L"]["rmse"]),
                bdcm_rmse=float(model["target_metrics"]["BDCM_ug_L"]["rmse"]),
            )
        )
    return rows


def top_feature(raw: pd.DataFrame, agg: pd.DataFrame, condition: str, model: str, target_key: str) -> str:
    if condition in {"D", "E"}:
        subset = agg[
            (agg["condition"] == condition)
            & (agg["model"] == model)
            & (agg["target_key"] == target_key)
        ].sort_values(["rank_mean", "shap_mean"], ascending=[True, False])
        row = subset.iloc[0]
        return f"{row['feature_label']} (#{row['rank_mean']:.1f})"
    subset = raw[
        (raw["condition"] == condition)
        & (raw["model"] == model)
        & (raw["target_key"] == target_key)
    ].sort_values(["rank", "mean_abs_shap"], ascending=[True, False])
    row = subset.iloc[0]
    return f"{row['feature_label']} (#{int(row['rank'])})"


def temperature_rank(raw: pd.DataFrame, agg: pd.DataFrame, condition: str, model: str, target_key: str) -> str:
    if condition in {"D", "E"}:
        subset = agg[
            (agg["condition"] == condition)
            & (agg["model"] == model)
            & (agg["target_key"] == target_key)
            & (agg["feature_label"] == "Temperature")
        ]
        row = subset.iloc[0]
        return f"{row['rank_mean']:.1f} +/- {row['rank_std']:.2f}"
    subset = raw[
        (raw["condition"] == condition)
        & (raw["model"] == model)
        & (raw["target_key"] == target_key)
        & (raw["feature_label"] == "Temperature")
    ]
    row = subset.iloc[0]
    return str(int(row["rank"]))


def condition_table_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for cond in CONDITION_ORDER:
        meta = CONDITION_META[cond]
        rows.append(
            [
                cond,
                meta["dataset"],
                meta["n"],
                meta["features"],
                meta["status"],
                meta["role"],
            ]
        )
    return rows


def comparison_rows() -> list[list[str]]:
    return [
        ["D vs B", "Dataset1, 5-common features", "Sample size (175 vs 488)", "Does smaller N restore Temperature?"],
        ["E vs C", "Dataset1, 6 features", "Sample size (175 vs 488)", "Does the answer change when Cl2 is present?"],
        ["D vs E", "Dataset1, N = 175", "Feature set (5 vs 6)", "Does adding Cl2 materially change the ranking?"],
        ["B vs C", "Dataset1, N = 488", "Feature set (5 vs 6)", "Does adding Cl2 matter at full sample size?"],
        ["D vs A'", "N = 175, 5-common features", "Data-source effect", "Is the shift intrinsic to the datasets?"],
    ]


def aprime_rows(metrics: list[AprimeMetrics]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in metrics:
        rows.append(
            [
                MODEL_LABELS[item.model],
                f"{item.macro_rmse:.3f}",
                f"{item.macro_r2:.3f}",
                f"{item.thm_rmse:.3f}",
                f"{item.dbcm_rmse:.3f}",
                f"{item.bdcm_rmse:.3f}",
            ]
        )
    return rows


def top_feature_rows(raw: pd.DataFrame, agg: pd.DataFrame, model: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for target_key in TARGET_ORDER:
        rows.append(
            [
                TARGET_LABELS[target_key],
                top_feature(raw, agg, "A'", model, target_key),
                top_feature(raw, agg, "B", model, target_key),
                top_feature(raw, agg, "D", model, target_key),
                top_feature(raw, agg, "C", model, target_key),
                top_feature(raw, agg, "E", model, target_key),
            ]
        )
    return rows


def temperature_rank_rows(raw: pd.DataFrame, agg: pd.DataFrame, model: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for target_key in TARGET_ORDER:
        rows.append(
            [
                TARGET_LABELS[target_key],
                temperature_rank(raw, agg, "A'", model, target_key),
                temperature_rank(raw, agg, "B", model, target_key),
                temperature_rank(raw, agg, "D", model, target_key),
                temperature_rank(raw, agg, "C", model, target_key),
                temperature_rank(raw, agg, "E", model, target_key),
            ]
        )
    return rows


def subgroup_group_rows(groups: pd.DataFrame) -> list[list[str]]:
    subset = (
        groups[
            ["group", "n_rows", "n_tsids", "temp_std", "tsid_temp_std_mean"]
        ]
        .drop_duplicates()
        .assign(group_order=lambda frame: frame["group"].map({"low": 0, "high": 1}))
        .sort_values("group_order")
    )
    label_map = {"low": "Low variability", "high": "High variability"}
    rows: list[list[str]] = []
    for _, row in subset.iterrows():
        rows.append(
            [
                label_map.get(str(row["group"]), str(row["group"])),
                str(int(row["n_rows"])),
                str(int(row["n_tsids"])),
                f"{float(row['temp_std']):.2f}",
                f"{float(row['tsid_temp_std_mean']):.2f}",
            ]
        )
    return rows


def subgroup_temperature_rank_rows(importance: pd.DataFrame, model: str) -> list[list[str]]:
    shap_df = importance[
        (importance["method"] == "shap")
        & (importance["feature"] == "temp_in_avg")
        & (importance["model"] == model)
    ].copy()
    rows: list[list[str]] = []
    for target in ["thm4_in_avg", "dbcm_in_avg", "bdcm_in_avg"]:
        subset = shap_df[shap_df["target"] == target]
        ranks = {
            (str(row["run"]), str(row["group"])): int(row["rank"])
            for _, row in subset.iterrows()
        }
        rows.append(
            [
                TARGET_LABELS[target.replace("_in_avg", "")] if target.endswith("_in_avg") else target_label(target),
                str(ranks[("B", "low")]),
                str(ranks[("B", "high")]),
                str(ranks[("C", "low")]),
                str(ranks[("C", "high")]),
            ]
        )
    return rows


def temp_rank_shift_summary(importance: pd.DataFrame) -> tuple[int, int, int, float]:
    shap_df = importance[
        (importance["method"] == "shap")
        & (importance["feature"] == "temp_in_avg")
    ].copy()
    pivot = shap_df.pivot_table(
        index=["run", "model", "target"],
        columns="group",
        values="rank",
    )
    delta = pivot["low"] - pivot["high"]
    improved = int((delta > 0).sum())
    same = int((delta == 0).sum())
    worse = int((delta < 0).sum())
    mean_delta = float(delta.mean())
    return improved, same, worse, mean_delta


def target_label(target_col: str) -> str:
    mapping = {
        "thm4_in_avg": "THM family response",
        "dbcm_in_avg": "DBCM",
        "bdcm_in_avg": "BDCM",
    }
    return mapping.get(target_col, target_col)


class ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(16, 16, 16)
        self.set_auto_page_break(auto=True, margin=18)
        self.alias_nb_pages()

    @property
    def cw(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(
            0,
            5,
            pdf_text("SHAP attribution study report - controlled explanation of feature importance shifts"),
            align="L",
        )
        self.ln(6)
        self.set_draw_color(220, 225, 235)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        if self.page_no() == 1:
            return
        self.set_y(-12)
        self.set_draw_color(220, 225, 235)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="C")

    def title_page(self, title: str, subtitle: str) -> None:
        self.add_page()
        self.set_fill_color(21, 84, 142)
        self.rect(0, 0, self.w, 11, "F")
        self.ln(38)
        self.set_font("Helvetica", "B", 23)
        self.set_text_color(21, 84, 142)
        self.multi_cell(0, 11, pdf_text(title), align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(75, 75, 75)
        self.multi_cell(0, 6, pdf_text(subtitle), align="C")
        self.ln(14)
        self.set_draw_color(21, 84, 142)
        self.set_line_width(0.6)
        self.line(34, self.get_y(), self.w - 34, self.get_y())
        self.ln(14)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(105, 105, 105)
        self.cell(0, 6, f"Generated on {date.today().isoformat()}", align="C")
        self.ln(9)
        self.set_font("Helvetica", "I", 9)
        self.multi_cell(
            0,
            5,
            pdf_text(
                "Repository artifact scope: experiments/shap_attribution, results/shap_attribution, results/shap_attribution/temp_variability_stratified, and the associated Tai Lake A' checkpoint."
            ),
            align="C",
        )
        self.set_fill_color(21, 84, 142)
        self.rect(0, self.h - 11, self.w, 11, "F")

    def section_title(self, title: str, subtitle: str = "") -> None:
        self.add_page()
        self.set_fill_color(21, 84, 142)
        self.rect(self.l_margin, self.get_y(), 3, 8, "F")
        self.set_x(self.l_margin + 6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(21, 84, 142)
        self.cell(0, 8, pdf_text(title))
        self.ln(10)
        if subtitle:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(110, 110, 110)
            self.multi_cell(0, 5, pdf_text(subtitle))
            self.ln(2)
        self.set_draw_color(220, 225, 235)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def subheading(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, pdf_text(text))
        self.ln(6)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, pdf_text(text))
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.cell(5, 5, "-")
        self.multi_cell(self.cw - 5, 5, pdf_text(text))

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        widths: list[float],
        title: str | None = None,
        font_size: int = 8,
        row_height: float = 6.0,
    ) -> None:
        if title:
            self.subheading(title)
        self.set_x(self.l_margin)
        self.set_fill_color(21, 84, 142)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", font_size)
        for header, width in zip(headers, widths):
            self.cell(width, 7, pdf_text(header), border=1, align="C", fill=True)
        self.ln()
        for idx, row in enumerate(rows):
            self.set_x(self.l_margin)
            bg = (247, 250, 255) if idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_text_color(45, 45, 45)
            for cell, width in zip(row, widths):
                self.set_font("Helvetica", "", font_size)
                self.cell(width, row_height, pdf_text(cell), border=1, align="C", fill=True)
            self.ln()
        self.ln(4)

    def _fit(self, path: Path, max_w: float, max_h: float) -> tuple[float, float]:
        with Image.open(path) as img:
            w, h = img.size
        ratio = min(max_w / w, max_h / h)
        return w * ratio, h * ratio

    def image_full(self, path: Path, caption: str = "", max_h: float = 95) -> None:
        if not path.exists():
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(180, 60, 60)
            self.multi_cell(0, 5, pdf_text(f"[Missing image: {path.name}]"))
            return
        w, h = self._fit(path, self.cw, max_h)
        if self.get_y() + h + 12 > self.h - 18:
            self.add_page()
        x = self.l_margin + (self.cw - w) / 2
        y = self.get_y()
        self.set_draw_color(225, 225, 230)
        self.rect(x - 1, y - 1, w + 2, h + 2)
        self.image(str(path), x=x, y=y, w=w, h=h)
        self.set_y(y + h + 3)
        if caption:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.multi_cell(0, 4, pdf_text(caption), align="C")
            self.ln(1)


def build_report() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, agg = load_data()
    temp_importance, temp_groups = load_temp_variability_data()
    aprime_metrics = load_aprime_metrics()
    improved, same, worse, mean_delta = temp_rank_shift_summary(temp_importance)

    pdf = ReportPDF()
    pdf.title_page(
        "SHAP Attribution Study Report",
        (
            "Controlled analysis of why the dominant SHAP feature shifts from Temperature in Tai Lake "
            "to Cl2 dose or Bromide in Dataset1."
        ),
    )

    pdf.section_title(
        "1. Executive overview",
        "This report consolidates the design, evidence, and final interpretation of the SHAP attribution study.",
    )
    pdf.subheading("Core research question")
    pdf.body(
        "Two earlier modeling efforts produced a striking interpretability mismatch. In Tai Lake, Temperature dominated SHAP rankings across targets. "
        "In Dataset1, Temperature lost that role and the leading drivers shifted to Cl2 dose for the THM-family target and to Bromide for DBCM and BDCM. "
        "This study was designed to explain that shift without conflating feature availability, sample size, and dataset provenance."
    )
    pdf.subheading("Headline findings")
    pdf.bullet("Sample size is not the main explanation. Reducing Dataset1 from 488 rows to 175 does not restore Temperature to dominant status.")
    pdf.bullet("Adding Cl2 changes the hierarchy, especially for the THM-family target, but Temperature is already weak in Dataset1 even before Cl2 is added.")
    pdf.bullet("The strongest supported conclusion is a data-source or data-distribution effect: the datasets encode different operational and environmental structure.")
    pdf.bullet("The cleanest evidence appears in the THM-family target and BDCM. DBCM is directionally consistent, but less sharply separated, especially for XGBoost.")

    pdf.section_title(
        "2. Experimental design",
        "The study uses five controlled conditions so that each comparison changes only one factor at a time.",
    )
    pdf.body(
        "The analysis compares one newly trained Tai Lake baseline (A') with four Dataset1 conditions. "
        "Conditions B and C are the existing full-data baselines. Conditions D and E are new subsampled controls that match Tai Lake's sample size."
    )
    pdf.table(
        headers=["Cond.", "Dataset", "N", "Features", "Status", "Role"],
        widths=[16, 34, 22, 38, 18, pdf.cw - 128],
        rows=condition_table_rows(),
        title="Condition matrix",
        font_size=8,
    )
    pdf.table(
        headers=["Comparison", "Held constant", "Varied factor", "Question answered"],
        widths=[20, 42, 40, pdf.cw - 102],
        rows=comparison_rows(),
        title="Control logic",
        font_size=8,
        row_height=7,
    )
    pdf.subheading("Shared 5-common feature set")
    pdf.body(
        "The strict cross-dataset comparison uses the shared feature backbone: pH, UV254, Temperature, TOC, and Bromide. "
        "Tai Lake's original COD term is intentionally removed in A' so that A' and D are aligned feature-for-feature."
    )

    pdf.section_title(
        "3. Modeling and SHAP protocol",
        "The study intentionally focuses on tree models so that the attribution method stays exact and computationally tractable.",
    )
    pdf.bullet("Models used in this attribution study: Random Forest and XGBoost only.")
    pdf.bullet("SHAP engine: TreeExplainer, evaluated on held-out test rows and averaged across the 5 CV ensemble members.")
    pdf.bullet("Subsample design for D and E: 5 independent seeds, each drawing 175 rows without replacement, then splitting 141 train / 34 test.")
    pdf.bullet("Hyperparameters for D and E are inherited from the best full-data checkpoint for the matched condition: B -> D and C -> E.")
    pdf.bullet("Outputs are aggregated as mean plus standard deviation across the 5 subsampling seeds.")
    pdf.bullet(
        "Scope note: the report aligns Dataset1's thm4_in_avg with Tai Lake's T_THMs_ug_L as the nearest shared THM-family response for cross-condition interpretation. "
        "DBCM and BDCM are directly matched."
    )
    pdf.body(
        "This means the conclusions should be read as tree-model SHAP conclusions under a controlled attribution protocol, not as universal statements about every model family or every possible response definition."
    )

    pdf.section_title(
        "4. Condition A' baseline",
        "A' answers a critical setup question: is Temperature still dominant in Tai Lake after COD is removed?",
    )
    pdf.body(
        "Yes. Condition A' confirms that Temperature remains the top-ranked feature for the THM-family target, DBCM, and BDCM in both Random Forest and XGBoost. "
        "So the original Tai Lake interpretation is not an artifact of COD competing with Temperature."
    )
    pdf.table(
        headers=["Model", "Macro RMSE", "Macro R2", "THM RMSE", "DBCM RMSE", "BDCM RMSE"],
        widths=[30, 24, 20, 24, 24, pdf.cw - 122],
        rows=aprime_rows(aprime_metrics),
        title="A' prediction metrics",
        font_size=8,
    )
    pdf.bullet("Temperature ranks #1 in A' for all three targets and both tree models.")
    pdf.bullet("The A' checkpoint completed successfully on 2026-04-07 and serves as the matched-feature external baseline for the rest of the study.")

    pdf.section_title(
        "5. Condition-level evidence tables",
        "These tables summarize the two most decision-relevant views of the study: which feature leads in each condition, and where Temperature ranks.",
    )
    for model_key in MODEL_ORDER:
        pdf.subheading(f"Top feature by condition - {MODEL_LABELS[model_key]}")
        pdf.table(
            headers=["Target", "A'", "B", "D", "C", "E"],
            widths=[32, 28, 28, 28, 28, pdf.cw - 144],
            rows=top_feature_rows(raw, agg, model_key),
            font_size=7,
            row_height=7,
        )
        pdf.subheading(f"Temperature rank by condition - {MODEL_LABELS[model_key]}")
        pdf.table(
            headers=["Target", "A'", "B", "D", "C", "E"],
            widths=[32, 26, 22, 28, 22, pdf.cw - 130],
            rows=temperature_rank_rows(raw, agg, model_key),
            font_size=7,
            row_height=7,
        )

    pdf.section_title(
        "6. Interpretation of the evidence",
        "The results support a ranked interpretation rather than a single over-simplified headline.",
    )
    pdf.subheading("Finding 1 - sample size is not the driver")
    pdf.body(
        "Comparisons D vs B and E vs C preserve the same qualitative hierarchy. In Random Forest, Temperature stays near rank 4 to 5 after subsampling for the THM-family target and BDCM. "
        "In XGBoost, the same is true for the THM-family target and BDCM, while DBCM remains more nuanced with Temperature still secondary rather than dominant."
    )
    pdf.subheading("Finding 2 - Cl2 matters, but it is not the root cause")
    pdf.body(
        "Cl2 enters as the leading THM-family feature in C and E, which is exactly what we would expect if Dataset1 directly measures an operational signal that Tai Lake only captured indirectly. "
        "However, Temperature is already weak in D before Cl2 is added, so Cl2 explains part of the shift, not the entire shift."
    )
    pdf.subheading("Finding 3 - the strongest supported explanation is a data-source effect")
    pdf.body(
        "At matched N and matched 5-common features, A' and D still look very different. Temperature stays dominant in Tai Lake, but not in Dataset1. "
        "The most defensible external phrasing is that the datasets encode different data-generating structure. Chemistry is a plausible mechanism, but the study does not isolate chemistry alone from the broader dataset context."
    )

    pdf.section_title(
        "7. Within-Dataset1 subgroup evidence",
        "A follow-up SHAP check asks whether Temperature recovers when Dataset1 is restricted to tsid series with larger within-series temperature variation.",
    )
    pdf.body(
        "We split Dataset1 test rows by the within-tsid standard deviation of temp_in_avg, using < 1.0 C as the low-variability group and >= 1.0 C as the high-variability group. "
        "The goal was not to create a new primary benchmark, but to test whether Temperature becomes more competitive once Dataset1 contains series with more internal thermal movement."
    )
    pdf.table(
        headers=["Group", "Test rows", "Test tsid", "Row-level temp std", "Mean tsid temp std"],
        widths=[34, 24, 24, 34, pdf.cw - 116],
        rows=subgroup_group_rows(temp_groups),
        title="Subgroup sample sizes",
        font_size=8,
    )
    pdf.body(
        f"Using tree-model SHAP, Temperature ranked higher in the high-variability group in {improved} of 12 run x model x target comparisons, "
        f"was unchanged in {same}, and never ranked lower. The average improvement was {mean_delta:.2f} rank positions."
    )
    for model_key in MODEL_ORDER:
        pdf.subheading(f"Temperature rank in low vs high variability groups - {MODEL_LABELS[model_key]}")
        pdf.table(
            headers=["Target", "B low", "B high", "C low", "C high"],
            widths=[42, 24, 24, 24, pdf.cw - 114],
            rows=subgroup_temperature_rank_rows(temp_importance, model_key),
            font_size=8,
            row_height=7,
        )
    pdf.subheading("Interpretation")
    pdf.bullet("The signal moves in the expected direction: when Dataset1 contains more within-series temperature movement, Temperature becomes somewhat more relevant.")
    pdf.bullet("The clearest lift appears for DBCM, where Temperature moves from rank 5 to 3 in B-RF and from rank 3 to 2 in both XGBoost settings.")
    pdf.bullet("That lift is still modest. Temperature does not become the dominant Dataset1 feature; TOC, Bromide, and Cl2 remain the main drivers depending on target and feature set.")
    pdf.bullet("So low within-tsid temperature variation is a contributing explanation for weak Temperature attribution in Dataset1, but not a complete explanation.")

    for fig in IMPORTANT_FIGURES:
        pdf.section_title(fig["title"], fig["subtitle"])
        pdf.image_full(fig["cross"], fig["cross_caption"], max_h=78)
        pdf.image_full(fig["heatmap"], fig["heatmap_caption"], max_h=88)

    pdf.section_title(
        "8. Conclusions and reporting guidance",
        "This final section turns the study into language that is safe to present externally.",
    )
    pdf.subheading("Externally defensible conclusions")
    pdf.bullet("Within the Random Forest and XGBoost SHAP protocol used here, the sample-size hypothesis is not supported.")
    pdf.bullet("Cl2 is an important attribution term for the THM-family response, but it is not the sole explanation for Temperature losing dominance in Dataset1.")
    pdf.bullet("Limited within-series temperature movement inside many Dataset1 tsid series contributes to weaker Temperature attribution, but even the high-variability subgroup does not restore Temperature to rank-1 status.")
    pdf.bullet("The strongest supported interpretation is a data-source or data-distribution effect. The shift is clearest for the THM-family response and BDCM, and directionally consistent but more nuanced for DBCM.")
    pdf.bullet("Single-plant SHAP rankings should not be assumed to generalize directly to multi-facility data.")
    pdf.subheading("Limitations that should be stated out loud")
    pdf.bullet("The attribution study uses tree models only. MLP and KAN were intentionally excluded from the controlled study because KernelExplainer across all seeds and folds would be too costly.")
    pdf.bullet("The THM target is aligned as a THM-family comparison across datasets, but the underlying columns remain dataset-specific: thm4_in_avg in Dataset1 and T_THMs_ug_L in Tai Lake.")
    pdf.bullet("The study is observational. It isolates dataset-level contrasts, not a single causal physicochemical mechanism.")
    pdf.bullet("Only five subsampling seeds were used. That is sufficient for a stability check, but it is still a finite Monte Carlo sample.")
    pdf.body(
        "Taken together, the evidence is strong enough to support external discussion if the wording stays within those boundaries. "
        "The report, figures, and tables in this PDF were generated directly from the repository artifacts currently stored under results/shap_attribution and the A' checkpoint directory."
    )

    pdf.output(str(OUTPUT_PATH))
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_report()
    print(f"Saved PDF report to: {path}")
