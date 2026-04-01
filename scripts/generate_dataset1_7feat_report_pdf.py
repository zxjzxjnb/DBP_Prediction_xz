"""Generate an English PDF report for the dataset1 7-feature study.

The report summarizes:
  - 7-feature experiment setup
  - formal performance across THM4 / BDCM / DBCM
  - best-model recommendations
  - SHAP findings with key figures for each target

Usage:
    python scripts/generate_dataset1_7feat_report_pdf.py
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
OUTPUT_PATH = OUTPUT_DIR / "dataset1_7feat_formal_report.pdf"

DATASET_PATH = PROJECT / "data" / "dataset1_dbp_formation_with_split.csv"
SHAP_DIR = PROJECT / "results" / "shap_analysis_dataset1_7feat_best"

RUN_DIRS = {
    "THM4": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_thm4_avg" / "20260331T214748Z",
    "BDCM": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_bdcm_avg" / "20260331T214748Z",
    "DBCM": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_dbcm_avg" / "20260331T225500Z",
}

BASELINE_REFERENCES = {
    "THM4": {
        "label": "6f + Cl2 scout",
        "path": PROJECT / "checkpoints" / "scout_dataset1_6feat_cl2d_avg" / "20260331T123111Z" / "metrics" / "model_comparison.json",
        "model_key": "rf",
    },
    "BDCM": {
        "label": "6f + Cl2 scout",
        "path": PROJECT / "checkpoints" / "scout_dataset1_6feat_cl2d_avg" / "20260331T123111Z" / "metrics" / "model_comparison.json",
        "model_key": "mlp",
    },
    "DBCM": {
        "label": "6f + contact formal",
        "path": PROJECT / "checkpoints" / "formal_dataset1_6feat_contact_time_dbcm_avg" / "20260331T214748Z" / "metrics" / "model_comparison.json",
        "model_key": "mlp",
    },
}

TARGET_KEYS = {
    "THM4": "thm4_in_avg",
    "BDCM": "bdcm_in_avg",
    "DBCM": "dbcm_in_avg",
}

MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "kan": "KAN",
}

FEATURE_ORDER = [
    "ph_in_avg",
    "uv_in_avg",
    "temp_in_avg",
    "toc_in_avg",
    "br_in_avg",
    "cl2d_in_avg",
    "time_sds_avg",
]

FEATURE_LABELS = {
    "ph_in_avg": "pH",
    "uv_in_avg": "UV254",
    "temp_in_avg": "Temperature",
    "toc_in_avg": "TOC",
    "br_in_avg": "Bromide",
    "cl2d_in_avg": "Cl2 dose",
    "time_sds_avg": "Contact time",
}


@dataclass(frozen=True)
class BestResult:
    target_label: str
    target_key: str
    model_name: str
    model_label: str
    rmse: float
    mae: float
    r2: float
    run_dir: Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_complete_case_count() -> tuple[int, int, int]:
    df = pd.read_csv(DATASET_PATH)
    keep = df.loc[:, ["split", *FEATURE_ORDER, *TARGET_KEYS.values()]].dropna(axis=0, how="any")
    train_rows = int((keep["split"] == "train").sum())
    test_rows = int((keep["split"] == "test").sum())
    return int(len(keep)), train_rows, test_rows


def load_result_tables() -> tuple[list[BestResult], dict[str, dict]]:
    best_results: list[BestResult] = []
    all_metrics: dict[str, dict] = {}
    for target_label, run_dir in RUN_DIRS.items():
        metrics = load_json(run_dir / "metrics" / "model_comparison.json")
        all_metrics[target_label] = metrics
        best_key = metrics["best_by_macro_rmse"]
        best_metrics = metrics["models"][best_key]["macro_test_metrics"]
        best_results.append(
            BestResult(
                target_label=target_label,
                target_key=TARGET_KEYS[target_label],
                model_name=best_key,
                model_label=MODEL_LABELS[best_key],
                rmse=float(best_metrics["rmse"]),
                mae=float(best_metrics["mae"]),
                r2=float(best_metrics["r2"]),
                run_dir=run_dir,
            )
        )
    return best_results, all_metrics


def load_shap_summary() -> pd.DataFrame:
    return pd.read_csv(SHAP_DIR / "mean_abs_shap_summary.csv")


def top_features_text(shap_df: pd.DataFrame, target_label: str, n: int = 3) -> str:
    subset = shap_df[shap_df["target_label"] == target_label].head(n)
    return ", ".join(
        f"{row.feature_label} ({row.mean_abs_shap:.3f})"
        for row in subset.itertuples()
    )


def image_path(target_key: str, name: str) -> Path:
    return SHAP_DIR / target_key / name


def load_baseline_comparison_rows(best_results: list[BestResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    best_by_target = {item.target_label: item for item in best_results}
    for target_label, meta in BASELINE_REFERENCES.items():
        baseline_metrics = load_json(meta["path"])
        baseline_vals = baseline_metrics["models"][meta["model_key"]]["macro_test_metrics"]
        current = best_by_target[target_label]
        delta_r2 = current.r2 - float(baseline_vals["r2"])
        delta_rmse = current.rmse - float(baseline_vals["rmse"])
        rows.append(
            [
                target_label,
                meta["label"],
                f"{baseline_vals['r2']:.4f}",
                f"{current.r2:.4f}",
                f"{delta_r2:+.4f}",
                f"{delta_rmse:+.3f}",
            ]
        )
    return rows


def load_shap_rank_rows(shap_df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for target_label in ["THM4", "BDCM", "DBCM"]:
        subset = shap_df[shap_df["target_label"] == target_label].reset_index(drop=True)
        labels = subset["feature_label"].tolist()
        contact_rank = labels.index("Contact time") + 1
        rows.append(
            [
                target_label,
                labels[0],
                labels[1],
                labels[2],
                str(contact_rank),
            ]
        )
    return rows


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
        self.set_text_color(105, 110, 120)
        self.cell(0, 5, "Dataset1 7-feature formal report", align="L")
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

    def title_page(self, title: str, subtitle: str, highlights: list[str] | None = None) -> None:
        self.add_page()
        self.set_fill_color(27, 74, 124)
        self.rect(0, 0, self.w, 10, "F")
        self.ln(34)
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(27, 74, 124)
        self.multi_cell(0, 12, title, align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 13)
        self.set_text_color(70, 70, 70)
        self.multi_cell(0, 7, subtitle, align="C")
        self.ln(10)
        self.set_draw_color(27, 74, 124)
        self.set_line_width(0.6)
        self.line(35, self.get_y(), self.w - 35, self.get_y())
        self.ln(10)
        if highlights:
            box_x = 28
            box_w = self.w - 56
            box_y = self.get_y()
            box_h = 10 + 7 * len(highlights)
            self.set_fill_color(244, 248, 253)
            self.set_draw_color(206, 220, 236)
            self.rect(box_x, box_y, box_w, box_h, "FD")
            self.set_xy(box_x + 6, box_y + 4)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(27, 74, 124)
            self.cell(0, 6, "Final 7-feature highlights")
            self.ln(8)
            self.set_x(box_x + 6)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(65, 65, 65)
            for item in highlights:
                self.set_x(box_x + 6)
                self.multi_cell(box_w - 12, 5, f"- {item}")
            self.set_y(box_y + box_h + 8)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Generated on {date.today().isoformat()}", align="C")
        self.set_fill_color(27, 74, 124)
        self.rect(0, self.h - 10, self.w, 10, "F")

    def section_title(self, title: str, subtitle: str = "", new_page: bool = True) -> None:
        if new_page:
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
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, text)
        self.ln(6)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, f"- {text}")
        self.ln(1)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float], title: str | None = None) -> None:
        if title:
            self.subheading(title)
        self.set_fill_color(27, 74, 124)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for header, width in zip(headers, widths):
            self.cell(width, 7, header, border=1, align="C", fill=True)
        self.ln()
        for idx, row in enumerate(rows):
            bg = (247, 250, 255) if idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*bg)
            self.set_text_color(45, 45, 45)
            self.set_font("Helvetica", "", 8)
            for cell, width in zip(row, widths):
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
            self.set_text_color(110, 110, 110)
            self.multi_cell(0, 4, caption, align="C")
        self.ln(2)


def build_pdf() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_results, all_metrics = load_result_tables()
    shap_df = load_shap_summary()
    complete_n, train_n, test_n = compute_complete_case_count()

    pdf = ReportPDF()
    pdf.title_page(
        "Dataset1 7-feature formal report",
        "Performance and SHAP interpretation for the 7-feature design: pH, UV254, temperature, TOC, bromide, Cl2 dose, and contact time.",
        highlights=[
            "Common feature backbone retained for THM4, BDCM, and DBCM.",
            "Final model choices: Random Forest for THM4 and BDCM, MLP for DBCM.",
            f"Complete-case sample count: {complete_n} rows ({train_n} train / {test_n} test).",
        ],
    )

    pdf.section_title(
        "Executive summary",
        "This report consolidates the final 7-feature formal runs and the target-specific SHAP interpretation built from the winning model of each target.",
    )
    pdf.body(
        "All three targets were analyzed with the same 7-feature backbone. The final recommendation is target-specific at the model level, but unified at the feature-set level: the 7-feature design is retained for THM4, BDCM, and DBCM."
    )
    pdf.bullet("THM4: Random Forest reached R2 = 0.8550 and delivered the best overall fit.")
    pdf.bullet("BDCM: Random Forest reached R2 = 0.8432 and remained the most stable formal winner.")
    pdf.bullet("DBCM: MLP reached R2 = 0.7134 and clearly outperformed the 7-feature XGBoost and KAN candidates.")
    pdf.bullet("Across SHAP results, Cl2 dose is consistently important, while contact time adds useful but secondary explanatory signal.")

    pdf.ln(2)
    pdf.section_title(
        "Experiment setup",
        "The same 7-feature representation was used for all targets. Missing rows were removed only when one of the selected features or the current target was unavailable.",
        new_page=False,
    )
    pdf.body(
        "Feature set: pH, UV254, Temperature, TOC, Bromide, Cl2 dose, and Contact time."
    )
    pdf.body(
        f"Complete-case sample count for the 7-feature setting: {complete_n} rows total ({train_n} train / {test_n} test)."
    )
    pdf.body(
        "Formal tuning used 5-fold cross-validation with stability-aware selection. Each target was optimized independently because the best-performing model family differed across THM4, BDCM, and DBCM."
    )
    pdf.table(
        headers=["Target", "Winning model", "RMSE", "MAE", "R2"],
        rows=[
            [r.target_label, r.model_label, f"{r.rmse:.3f}", f"{r.mae:.3f}", f"{r.r2:.4f}"]
            for r in best_results
        ],
        widths=[28, 45, 28, 28, 24],
        title="Final recommendations",
    )

    pdf.section_title(
        "Formal results",
        "The first table reports the complete formal test-set metrics under the 7-feature configuration. The second table compares each final winner against its strongest earlier reference setting.",
    )
    formal_rows = []
    for target_label in ["THM4", "BDCM", "DBCM"]:
        for model_key in ["rf", "xgb", "mlp", "kan"]:
            vals = all_metrics[target_label]["models"][model_key]["macro_test_metrics"]
            formal_rows.append(
                [
                    target_label,
                    MODEL_LABELS[model_key],
                    f"{vals['rmse']:.3f}",
                    f"{vals['mae']:.3f}",
                    f"{vals['r2']:.4f}",
                ]
            )
    pdf.table(
        headers=["Target", "Model", "RMSE", "MAE", "R2"],
        rows=formal_rows,
        widths=[22, 44, 24, 24, 22],
        title="All formal metrics under the 7-feature setting",
    )
    pdf.table(
        headers=["Target", "Prior best", "Prior R2", "7f R2", "Delta R2", "Delta RMSE"],
        rows=load_baseline_comparison_rows(best_results),
        widths=[18, 44, 20, 18, 22, 24],
        title="Improvement versus the strongest earlier reference",
    )
    pdf.subheading("Performance interpretation")
    pdf.bullet("Random Forest remained the most reliable family for THM4 and BDCM, which suggests that both targets benefit from robust non-linear partitioning under the expanded 7-feature design.")
    pdf.bullet("DBCM behaved differently: its best formal result came from MLP rather than tree-based models, indicating a more distributed and smoother response surface across the selected predictors.")
    pdf.bullet("KAN did not win any of the three final formal runs, so the final 7-feature recommendation now favors simpler and more stable model families for deployment or reporting.")

    pdf.section_title(
        "SHAP overview",
        "The overview below combines the cross-target feature-importance picture with the most important model-selection takeaways.",
    )
    pdf.bullet(
        "THM4 benefited the most from Cl2 dose. The 7-feature Random Forest improved over the earlier 6-feature + Cl2 scout winner from R2 = 0.8275 to R2 = 0.8550."
    )
    pdf.bullet(
        "BDCM also preferred the 7-feature setting, with Random Forest reaching R2 = 0.8432 and slightly exceeding the earlier best 6-feature + Cl2 scout result (R2 = 0.8270)."
    )
    pdf.bullet(
        "DBCM showed the clearest gain from fully unifying around 7 features: the final MLP achieved R2 = 0.7134, improving over the previous 6-feature + contact-time formal MLP result of R2 = 0.6355."
    )
    pdf.bullet(
        "The 7-feature DBCM run is especially important because scout had favored XGBoost, but formal tuning shifted the final winner to MLP, indicating better generalization after full validation."
    )
    pdf.image_full(
        SHAP_DIR / "cross_target_importance.png",
        "Cross-target mean absolute SHAP values for the three selected 7-feature models.",
        max_h=90,
    )
    pdf.table(
        headers=["Target", "Top 1", "Top 2", "Top 3", "Contact rank"],
        rows=load_shap_rank_rows(shap_df),
        widths=[20, 36, 36, 36, 24],
        title="Top SHAP ranks under the selected 7-feature models",
    )
    pdf.bullet("Final recommendation: retain 7 features as the common dataset1 design backbone.")
    pdf.bullet("Model choice remains target-specific: Random Forest for THM4 and BDCM, MLP for DBCM.")
    pdf.bullet("Cl2 dose is the strongest newly added explanatory variable across the full 7-feature study.")
    pdf.bullet("Contact time is consistently non-zero in SHAP, but it is not a top-3 driver for any of the three targets.")

    for target_label in ["THM4", "BDCM", "DBCM"]:
        target_key = TARGET_KEYS[target_label]
        result = next(item for item in best_results if item.target_label == target_label)
        pdf.section_title(
            f"{target_label} interpretation",
            f"Best model: {result.model_label}. This page combines the ranking of mean absolute SHAP values with the SHAP beeswarm pattern.",
        )
        pdf.body(
            f"Top SHAP drivers for {target_label}: {top_features_text(shap_df, target_label)}."
        )
        if target_label == "THM4":
            pdf.body(
                "Interpretation: THM4 is primarily driven by Cl2 dose and TOC, while bromide provides a secondary but still meaningful contribution. Contact time is helpful, but it is not the dominant source of variance."
            )
        elif target_label == "BDCM":
            pdf.body(
                "Interpretation: BDCM is bromide-led, with Cl2 dose providing the second-largest incremental effect. UV254 and TOC stay relevant, which suggests that precursor quality still matters after disinfection-related variables are included."
            )
        else:
            pdf.body(
                "Interpretation: DBCM is more evenly distributed across bromide, UV254, Cl2 dose, temperature, and TOC than the other targets. Contact time contributes positively, but the strongest 7-feature DBCM signal still comes from bromide and precursor-related variables."
            )
        pdf.image_full(
            image_path(target_key, "bar_importance.png"),
            f"{target_label}: mean absolute SHAP values.",
            max_h=78,
        )
        pdf.image_full(
            image_path(target_key, "summary_beeswarm.png"),
            f"{target_label}: SHAP beeswarm summary.",
            max_h=95,
        )

    pdf.output(str(OUTPUT_PATH))
    return OUTPUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(path)
