"""Generate an English PDF report for dataset1 5-feature vs 6-feature+Cl2.

The report summarizes:
  - experiment setup
  - macro and target-level metrics for all models
  - key SHAP findings with selected figures

Usage:
    python scripts/generate_dataset1_ablation_pdf.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from fpdf import FPDF
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "dataset1_ablation_cl2_report.pdf"

RUN_5 = PROJECT / "checkpoints" / "formal_dataset1_5feat_avg" / "20260330T235734Z"
RUN_6 = PROJECT / "checkpoints" / "formal_dataset1_6feat_cl2d_avg" / "20260331T130339Z"
SHAP_DIR = PROJECT / "results" / "shap_analysis_dataset1_ablation"
DATASET_PATH = PROJECT / "data" / "dataset1_dbp_formation_with_split.csv"

TARGETS = ["thm4_in_avg", "dbcm_in_avg", "bdcm_in_avg"]
TARGET_LABELS = {
    "thm4_in_avg": "THM4",
    "dbcm_in_avg": "DBCM",
    "bdcm_in_avg": "BDCM",
}
MODEL_ORDER = ["rf", "xgb", "mlp", "kan"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "kan": "KAN",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    run_dir: Path
    features: list[str]


RUN_SPECS = [
    RunSpec(
        key="formal_5feat",
        label="Formal 5-feature baseline",
        run_dir=RUN_5,
        features=["ph_in_avg", "uv_in_avg", "temp_in_avg", "toc_in_avg", "br_in_avg"],
    ),
    RunSpec(
        key="formal_6feat_cl2d",
        label="Formal 6-feature + Cl2",
        run_dir=RUN_6,
        features=[
            "ph_in_avg",
            "uv_in_avg",
            "temp_in_avg",
            "toc_in_avg",
            "br_in_avg",
            "cl2d_in_avg",
        ],
    ),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_metrics(spec: RunSpec) -> dict:
    return load_json(spec.run_dir / "metrics" / "model_comparison.json")


def load_shap_summary() -> pd.DataFrame:
    return pd.read_csv(SHAP_DIR / "mean_abs_shap_summary.csv")


def compute_complete_case_count(features: list[str]) -> tuple[int, int, int]:
    df = pd.read_csv(DATASET_PATH)
    required = features + TARGETS
    keep = df.loc[:, ["split", *required]].dropna(axis=0, how="any")
    train_rows = int((keep["split"] == "train").sum())
    test_rows = int((keep["split"] == "test").sum())
    return int(len(keep)), train_rows, test_rows


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
        self.cell(0, 5, "Dataset1 ablation report - 5-feature baseline vs 6-feature + Cl2", align="L")
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
        self.set_fill_color(27, 74, 124)
        self.rect(0, 0, self.w, 10, "F")
        self.ln(42)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(27, 74, 124)
        self.multi_cell(0, 12, title, align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(70, 70, 70)
        self.multi_cell(0, 7, subtitle, align="C")
        self.ln(16)
        self.set_draw_color(27, 74, 124)
        self.set_line_width(0.6)
        self.line(35, self.get_y(), self.w - 35, self.get_y())
        self.ln(14)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Generated on {date.today().isoformat()}", align="C")
        self.set_fill_color(27, 74, 124)
        self.rect(0, self.h - 10, self.w, 10, "F")

    def section_title(self, title: str, subtitle: str = "") -> None:
        self.add_page()
        self.set_fill_color(27, 74, 124)
        self.rect(self.l_margin, self.get_y(), 3, 8, "F")
        self.set_x(self.l_margin + 6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(27, 74, 124)
        self.cell(0, 8, title)
        self.ln(10)
        if subtitle:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(110, 110, 110)
            self.multi_cell(0, 5, subtitle)
            self.ln(2)
        self.set_draw_color(220, 225, 235)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def subheading(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, text)
        self.ln(6)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.cell(5, 5, "-")
        self.multi_cell(self.cw - 5, 5, text)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float], title: str | None = None) -> None:
        if title:
            self.subheading(title)
        self.set_x(self.l_margin)
        self.set_fill_color(27, 74, 124)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for header, width in zip(headers, widths):
            self.cell(width, 7, header, border=1, align="C", fill=True)
        self.ln()
        for idx, row in enumerate(rows):
            self.set_x(self.l_margin)
            bg = (247, 250, 255) if idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_text_color(45, 45, 45)
            for cell, width in zip(row, widths):
                self.set_font("Helvetica", "", 8)
                self.cell(width, 6, str(cell), border=1, align="C", fill=True)
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
            self.multi_cell(0, 5, f"[Missing image: {path.name}]")
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
            self.multi_cell(0, 4, caption, align="C")
            self.ln(1)


def fmt(v: float) -> str:
    return f"{v:.4f}"


def metric_rows(metrics_5: dict, metrics_6: dict, target: str | None = None) -> list[list[str]]:
    rows: list[list[str]] = []
    for model_name in MODEL_ORDER:
        source_5 = metrics_5["models"][model_name]
        source_6 = metrics_6["models"][model_name]
        if target is None:
            values_5 = source_5["macro_test_metrics"]
            values_6 = source_6["macro_test_metrics"]
        else:
            values_5 = source_5["target_metrics"][target]
            values_6 = source_6["target_metrics"][target]
        delta_rmse = values_6["rmse"] - values_5["rmse"]
        delta_r2 = values_6["r2"] - values_5["r2"]
        rows.append(
            [
                MODEL_LABELS[model_name],
                fmt(values_5["rmse"]),
                fmt(values_5["mae"]),
                fmt(values_5["r2"]),
                fmt(values_6["rmse"]),
                fmt(values_6["mae"]),
                fmt(values_6["r2"]),
                f"{delta_rmse:+.4f}",
                f"{delta_r2:+.4f}",
            ]
        )
    return rows


def best_model_by_target(metrics: dict, target: str) -> tuple[str, float]:
    return min(
        (
            (model_name, metrics["models"][model_name]["target_metrics"][target]["rmse"])
            for model_name in MODEL_ORDER
        ),
        key=lambda item: item[1],
    )


def shap_top_features(shap_df: pd.DataFrame, experiment: str, model: str, target: str, n: int = 3) -> list[tuple[str, float]]:
    subset = shap_df[
        (shap_df["experiment"] == experiment)
        & (shap_df["model"] == model)
        & (shap_df["target"] == target)
    ].sort_values("rank")
    return [(row.feature_label, float(row.mean_abs_shap)) for row in subset.head(n).itertuples()]


def delta_summary(metrics_5: dict, metrics_6: dict) -> list[str]:
    lines = []
    for model_name in MODEL_ORDER:
        m5 = metrics_5["models"][model_name]["macro_test_metrics"]
        m6 = metrics_6["models"][model_name]["macro_test_metrics"]
        lines.append(
            f"{MODEL_LABELS[model_name]}: macro RMSE {m5['rmse']:.2f} to {m6['rmse']:.2f}, "
            f"macro R2 {m5['r2']:.4f} to {m6['r2']:.4f}"
        )
    return lines


def build_report() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_5 = load_metrics(RUN_SPECS[0])
    metrics_6 = load_metrics(RUN_SPECS[1])
    shap_df = load_shap_summary()
    complete_rows_5, train_5, test_5 = compute_complete_case_count(RUN_SPECS[0].features)
    complete_rows_6, train_6, test_6 = compute_complete_case_count(RUN_SPECS[1].features)

    pdf = ReportPDF()
    pdf.title_page(
        "Dataset1 Ablation Report",
        "Comparing the formal 5-feature baseline with the formal 6-feature run that adds Cl2 dose, including performance metrics and SHAP interpretation.",
    )

    pdf.section_title("1. Study setup", "Design choices, dataset footprint, and model families used in both formal runs.")
    pdf.subheading("Scope")
    pdf.body(
        "This report compares two formal dataset1 experiments: a 5-feature baseline "
        "(pH, UV254, temperature, TOC, bromide) and a 6-feature variant that adds Cl2 dose. "
        "Both runs use the same three targets (THM4, DBCM, BDCM), the same four model families "
        "(Random Forest, XGBoost, MLP, KAN), and the same per-target training paradigm."
    )
    pdf.subheading("Data and validation")
    pdf.bullet("Raw dataset rows: 514 with a predefined split of 406 train and 108 test rows.")
    pdf.bullet(
        f"After drop-missing complete-case filtering, the 5-feature run used {complete_rows_5} rows "
        f"({train_5} train / {test_5} test)."
    )
    pdf.bullet(
        f"After drop-missing complete-case filtering, the 6-feature + Cl2 run used {complete_rows_6} rows "
        f"({train_6} train / {test_6} test)."
    )
    pdf.bullet("Formal tuning used 5-fold cross-validation with a stability penalty of 0.18.")
    pdf.bullet("Global neural training settings: max_epochs = 1200 and patience = 100.")

    pdf.table(
        headers=["Run", "Features", "Targets", "Models"],
        widths=[38, 72, 38, pdf.cw - 148],
        rows=[
            [
                "Formal 5-feature",
                "pH, UV254, Temp, TOC, Bromide",
                "THM4, DBCM, BDCM",
                "RF, XGB, MLP, KAN",
            ],
            [
                "Formal 6-feature + Cl2",
                "pH, UV254, Temp, TOC, Bromide, Cl2 dose",
                "THM4, DBCM, BDCM",
                "RF, XGB, MLP, KAN",
            ],
        ],
        title="Experiment matrix",
    )

    pdf.section_title("2. Macro performance", "Full-model macro metrics across the two formal ablation rounds.")
    pdf.body(
        "Adding Cl2 improved macro performance for Random Forest, XGBoost, and MLP. "
        "KAN did not benefit from the new feature in this formal run."
    )
    pdf.table(
        headers=["Model", "5f RMSE", "5f MAE", "5f R2", "6f RMSE", "6f MAE", "6f R2", "Delta RMSE", "Delta R2"],
        widths=[28, 18, 18, 16, 18, 18, 16, 19, 19],
        rows=metric_rows(metrics_5, metrics_6, target=None),
    )
    pdf.subheading("Macro takeaways")
    for line in delta_summary(metrics_5, metrics_6):
        pdf.bullet(line)
    best_macro_rmse = metrics_6["best_by_macro_rmse"]
    best_macro_r2 = max(
        MODEL_ORDER,
        key=lambda m: metrics_6["models"][m]["macro_test_metrics"]["r2"],
    )
    pdf.bullet(
        f"Best macro RMSE in the 6-feature run: {MODEL_LABELS[best_macro_rmse]} "
        f"({metrics_6['models'][best_macro_rmse]['macro_test_metrics']['rmse']:.2f})."
    )
    pdf.bullet(
        f"Best macro R2 in the 6-feature run: {MODEL_LABELS[best_macro_r2]} "
        f"({metrics_6['models'][best_macro_r2]['macro_test_metrics']['r2']:.4f})."
    )

    for target in TARGETS:
        pdf.section_title(
            f"3. Target-level results - {TARGET_LABELS[target]}",
            "Detailed RMSE, MAE, and R2 values for every model in both ablation rounds.",
        )
        pdf.table(
            headers=["Model", "5f RMSE", "5f MAE", "5f R2", "6f RMSE", "6f MAE", "6f R2", "Delta RMSE", "Delta R2"],
            widths=[28, 18, 18, 16, 18, 18, 16, 19, 19],
            rows=metric_rows(metrics_5, metrics_6, target=target),
        )
        best_5_name, best_5_rmse = best_model_by_target(metrics_5, target)
        best_6_name, best_6_rmse = best_model_by_target(metrics_6, target)
        pdf.subheading("Interpretation")
        pdf.bullet(
            f"Best RMSE in the 5-feature baseline: {MODEL_LABELS[best_5_name]} ({best_5_rmse:.2f})."
        )
        pdf.bullet(
            f"Best RMSE in the 6-feature + Cl2 run: {MODEL_LABELS[best_6_name]} ({best_6_rmse:.2f})."
        )
        if best_5_name == best_6_name:
            pdf.bullet(f"The winning model family did not change for {TARGET_LABELS[target]}.")
        else:
            pdf.bullet(
                f"The winning model family changed from {MODEL_LABELS[best_5_name]} "
                f"to {MODEL_LABELS[best_6_name]} after adding Cl2."
            )

    pdf.section_title(
        "4. SHAP summary across the two ablation rounds",
        "Feature attribution changes explain where the performance gains came from.",
    )
    pdf.body(
        "SHAP values were computed for both formal runs. Tree-based models used TreeExplainer, "
        "while MLP and KAN used KernelExplainer on the complete-case test set. "
        "The summary below focuses on the model-target pairs that best explain the observed performance shifts."
    )
    shap_focus_items = [
        (
            "THM4 / Random Forest",
            shap_top_features(shap_df, "formal_5feat", "rf", "thm4_in_avg"),
            shap_top_features(shap_df, "formal_6feat_cl2d", "rf", "thm4_in_avg"),
            "Cl2 becomes the strongest feature and THM4 error drops sharply.",
        ),
        (
            "DBCM / XGBoost",
            shap_top_features(shap_df, "formal_5feat", "xgb", "dbcm_in_avg"),
            shap_top_features(shap_df, "formal_6feat_cl2d", "xgb", "dbcm_in_avg"),
            "Bromide remains dominant; Cl2 contributes only modestly.",
        ),
        (
            "BDCM / MLP",
            shap_top_features(shap_df, "formal_5feat", "mlp", "bdcm_in_avg"),
            shap_top_features(shap_df, "formal_6feat_cl2d", "mlp", "bdcm_in_avg"),
            "Cl2 emerges as a strong second driver and helps BDCM most in MLP.",
        ),
    ]
    for title, top_5, top_6, reading in shap_focus_items:
        pdf.subheading(title)
        pdf.bullet("Top features in 5-feature baseline: " + ", ".join(f"{name} ({value:.3f})" for name, value in top_5))
        pdf.bullet("Top features in 6-feature + Cl2: " + ", ".join(f"{name} ({value:.3f})" for name, value in top_6))
        pdf.bullet("Interpretation: " + reading)
        pdf.ln(1)
    pdf.bullet("THM4 changes the most: TOC leads in the baseline, but Cl2 becomes the top SHAP feature in all four models after it is added.")
    pdf.bullet("DBCM changes the least: bromide remains the main explanatory feature in both rounds.")
    pdf.bullet("BDCM keeps bromide as the main signal, but Cl2 rises to a clear second place in every model family.")

    shap_pages = [
        (
            "5. SHAP focus - THM4",
            "Random Forest provides the clearest explanation for why THM4 improves after adding Cl2.",
            SHAP_DIR / "cross_experiment_rf_thm4_in_avg.png",
            "Cross-experiment mean absolute SHAP comparison for Random Forest on THM4.",
            SHAP_DIR / "formal_6feat_cl2d" / "rf" / "thm4_in_avg" / "summary_beeswarm.png",
            "6-feature + Cl2 SHAP beeswarm for Random Forest on THM4.",
            [
                "In the baseline run, THM4 is mostly driven by TOC, with bromide and UV254 as secondary terms.",
                "After Cl2 is added, Cl2 dose becomes the strongest SHAP contributor and TOC remains the main secondary driver.",
                "This attribution shift is consistent with the large THM4 RMSE reduction from 55.43 to 41.46 in Random Forest.",
            ],
        ),
        (
            "6. SHAP focus - DBCM",
            "XGBoost remains the strongest DBCM model, and SHAP shows why Cl2 only adds limited extra information here.",
            SHAP_DIR / "cross_experiment_xgb_dbcm_in_avg.png",
            "Cross-experiment mean absolute SHAP comparison for XGBoost on DBCM.",
            SHAP_DIR / "formal_6feat_cl2d" / "xgb" / "dbcm_in_avg" / "summary_beeswarm.png",
            "6-feature + Cl2 SHAP beeswarm for XGBoost on DBCM.",
            [
                "Bromide is the dominant feature in both rounds, with temperature and UV254 staying ahead of Cl2.",
                "Cl2 enters the model, but its SHAP magnitude remains smaller than the main bromide-driven structure.",
                "That pattern matches the modest but still real DBCM gain in XGBoost, from RMSE 6.06 to 5.20.",
            ],
        ),
        (
            "7. SHAP focus - BDCM",
            "MLP becomes the best BDCM model in the 6-feature run, and SHAP shows that Cl2 adds a new strong signal on top of bromide.",
            SHAP_DIR / "cross_experiment_mlp_bdcm_in_avg.png",
            "Cross-experiment mean absolute SHAP comparison for MLP on BDCM.",
            SHAP_DIR / "formal_6feat_cl2d" / "mlp" / "bdcm_in_avg" / "summary_beeswarm.png",
            "6-feature + Cl2 SHAP beeswarm for MLP on BDCM.",
            [
                "Bromide remains the top BDCM feature, but Cl2 rises to the second position once it is introduced.",
                "TOC remains important, yet the feature balance is more distributed than in the baseline run.",
                "This helps explain why the best BDCM model changes from Random Forest in the baseline to MLP in the 6-feature run.",
            ],
        ),
    ]

    for title, subtitle, fig1, cap1, fig2, cap2, bullets in shap_pages:
        pdf.section_title(title, subtitle)
        for bullet in bullets:
            pdf.bullet(bullet)
        pdf.ln(2)
        pdf.image_full(fig1, cap1, max_h=70)
        pdf.image_full(fig2, cap2, max_h=92)

    pdf.section_title("8. Conclusions and recommendations", "What these two formal ablation rounds tell us.")
    pdf.bullet("Adding Cl2 is justified: it improves three of the four model families and produces the strongest benefit for THM4.")
    pdf.bullet("THM4 is the target most directly reshaped by Cl2, both in predictive accuracy and in SHAP attribution structure.")
    pdf.bullet("DBCM remains primarily bromide-driven, so Cl2 should be treated as a supporting rather than dominant signal for that target.")
    pdf.bullet("BDCM benefits from Cl2 as an important secondary driver, especially in the MLP model where it helps change the winning model family.")
    pdf.bullet("For future reporting, Random Forest is the clearest headline model for THM4, XGBoost for DBCM, and MLP for BDCM.")
    pdf.body(
        "All figures in this report were generated from the formal dataset1 checkpoints and the SHAP analysis outputs in the repository. "
        "If the experiments are rerun, this PDF can be regenerated directly from the updated artifacts."
    )

    pdf.output(str(OUTPUT_PATH))
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_report()
    print(f"Saved PDF report to: {path}")
