"""Member 5 evaluation and visualization pipeline.

Run from any working directory with:
    python evaluation.py

The script reads existing outputs from Members 2, 3, and 4. It does not
re-run or modify their processing pipelines or source results.

Important: BSI + Otsu masks are used as automatic pseudo-labels (weak
reference masks) to compare against the Texture + Morphology refined masks.
IoU/Dice/Precision/Recall therefore measure agreement with the baseline
method only; they are NOT independent segmentation-accuracy scores because
no manually annotated Ground Truth masks are available.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
except ImportError:  # Mask comparison figure is optional without rasterio.
    rasterio = None


# ---------------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

BSI_SUMMARY = ROOT / "Segmentation" / "Results_BSI_Otsu" / "BSI_Otsu_Summary.csv"
REFINEMENT_SUMMARY = ROOT / "Texture Morphology Results" / "summary.csv"

TEMPORAL_ROOT = ROOT / "Temporal Change Results" / "Temporal_Change_Results"
ANNUAL_AREA_FILE = TEMPORAL_ROOT / "tables" / "annual_sand_area.csv"
LONG_TERM_FILE = TEMPORAL_ROOT / "tables" / "change_2019_2026.csv"
TEMPORAL_STATS_FILE = TEMPORAL_ROOT / "tables" / "temporal_change_statistics.csv"
PROVENANCE_FILE = TEMPORAL_ROOT / "analysis_provenance.json"

BSI_MASK_DIR = ROOT / "Segmentation" / "Results_BSI_Otsu"
REFINED_MASK_DIR = ROOT / "Texture Morphology Results" / "masks"
PREVIEW_DIR = ROOT / "Texture Morphology Results" / "previews"
OUTPUT_DIR = ROOT / "Evaluation Results"
FIGURE_DIR = OUTPUT_DIR / "figures"
SELECTED_PREVIEW_DIR = OUTPUT_DIR / "selected_previews"
PSEUDO_DIFFERENCE_DIR = OUTPUT_DIR / "pseudo_label_difference_masks"

# Numerical tolerances used only for validation of saved tables.
AREA_TOLERANCE_KM2 = 1e-5
PERCENTAGE_TOLERANCE = 0.02  # percentage points; allows for rounded summary fractions.
FRACTION_TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# 2. GENERAL HELPERS
# ---------------------------------------------------------------------------
def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}:\n{path}\n\n"
            "Check that the repository folders and CSV files have been extracted "
            "and that this script is saved in the repository root."
        )


def require_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )


def numeric_columns(df: pd.DataFrame, columns: list[str], file_label: str) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any():
            bad_count = int(df[column].isna().sum())
            raise ValueError(
                f"{file_label}: column '{column}' contains {bad_count} missing/non-numeric values."
            )


def max_abs_difference(left: pd.Series, right: pd.Series) -> float:
    values = (pd.to_numeric(left) - pd.to_numeric(right)).abs().to_numpy(dtype=float)
    return float(np.nanmax(values)) if values.size else 0.0


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: str,
    status_override: str | None = None,
) -> None:
    status = status_override or ("PASS" if passed else "CHECK")
    checks.append({"check": name, "status": status, "details": details})


def save_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUTPUT_DIR / filename, index=False)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3. LOAD AND MATCH MEMBER 2 + MEMBER 3 RESULTS
# ---------------------------------------------------------------------------
def load_method_comparison(checks: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(BSI_SUMMARY, "Member 2 BSI-Otsu summary")
    require_file(REFINEMENT_SUMMARY, "Member 3 texture-morphology summary")

    bsi = pd.read_csv(BSI_SUMMARY)
    refined = pd.read_csv(REFINEMENT_SUMMARY)

    require_columns(
        bsi,
        ["Image", "Year", "Potential_Sand_Percentage"],
        "BSI_Otsu_Summary.csv",
    )
    require_columns(
        refined,
        [
            "location", "year", "segmented_fraction", "removed_texture",
            "removed_salt_flat", "refined_fraction", "refined_area_km2",
            "patches_before", "patches_after",
        ],
        "Texture Morphology Results/summary.csv",
    )

    # BSI filenames use: <location>_<year>_processed.tif
    bsi_parts = bsi["Image"].astype(str).str.extract(
        r"^(?P<bsi_location>.+)_(?P<year>20\d{2})_processed\.tif$"
    )
    if bsi_parts.isna().any().any():
        examples = bsi.loc[bsi_parts.isna().any(axis=1), "Image"].head(5).tolist()
        raise ValueError(f"Could not parse location/year from BSI filenames: {examples}")

    bsi = bsi.copy()
    bsi["bsi_location"] = bsi_parts["bsi_location"]
    bsi["location_id"] = bsi["bsi_location"].str.extract(r"^(L\d+)", expand=False)
    bsi["year"] = pd.to_numeric(bsi_parts["year"], errors="raise").astype(int)

    refined = refined.copy()
    refined["location"] = refined["location"].astype(str).str.strip()
    refined["location_id"] = refined["location"].str.extract(r"^(L\d+)", expand=False)
    refined["year"] = pd.to_numeric(refined["year"], errors="raise").astype(int)

    if bsi["location_id"].isna().any() or refined["location_id"].isna().any():
        raise ValueError("A location ID could not be extracted from one or more summary rows.")

    duplicate_bsi = int(bsi.duplicated(["location_id", "year"]).sum())
    duplicate_refined = int(refined.duplicated(["location_id", "year"]).sum())
    add_check(checks, "No duplicate BSI location-year rows", duplicate_bsi == 0, f"Duplicates: {duplicate_bsi}")
    add_check(checks, "No duplicate refined location-year rows", duplicate_refined == 0, f"Duplicates: {duplicate_refined}")
    if duplicate_bsi or duplicate_refined:
        raise ValueError("Duplicate location-year records found. Resolve these before comparison.")

    results = bsi.merge(
        refined,
        on=["location_id", "year"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_bsi", "_refined"),
    )

    match_counts = results["_merge"].value_counts().to_dict()
    unmatched = int((results["_merge"] != "both").sum())
    add_check(
        checks,
        "All BSI and refined images matched",
        unmatched == 0,
        f"Matched: {match_counts.get('both', 0)}; left only: {match_counts.get('left_only', 0)}; right only: {match_counts.get('right_only', 0)}",
    )
    if unmatched:
        raise ValueError(f"{unmatched} BSI/refinement records could not be matched by location and year.")

    # Confirm that the location text itself matches, not only the L-number.
    different_names = results["bsi_location"].astype(str) != results["location"].astype(str)
    add_check(
        checks,
        "Location names agree between methods",
        not different_names.any(),
        f"Different location names: {int(different_names.sum())}",
    )
    if different_names.any():
        raise ValueError("Some BSI and refined rows have the same location ID/year but different location names.")

    numeric_columns(
        results,
        [
            "Potential_Sand_Percentage", "segmented_fraction", "refined_fraction",
            "refined_area_km2", "patches_before", "patches_after",
            "removed_texture", "removed_salt_flat",
        ],
        "Matched method results",
    )

    # The initial BSI percentage should match the segmented fraction saved by Member 3.
    initial_difference = max_abs_difference(
        results["Potential_Sand_Percentage"] / 100.0,
        results["segmented_fraction"],
    )
    add_check(
        checks,
        "Initial segmentation values agree across summaries",
        initial_difference <= FRACTION_TOLERANCE,
        f"Maximum absolute difference in fraction: {initial_difference:.8f}",
    )

    results["initial_sand_pct"] = results["Potential_Sand_Percentage"]
    results["refined_sand_pct"] = results["refined_fraction"] * 100.0
    results["change_pp"] = results["refined_sand_pct"] - results["initial_sand_pct"]

    per_image = results[
        [
            "location_id", "location", "year", "Image",
            "initial_sand_pct", "refined_sand_pct", "change_pp",
            "removed_texture", "removed_salt_flat",
            "patches_before", "patches_after", "refined_area_km2",
        ]
    ].copy()

    yearly = results.groupby("year", as_index=False).agg(
        image_count=("location_id", "count"),
        initial_mean_pct=("initial_sand_pct", "mean"),
        refined_mean_pct=("refined_sand_pct", "mean"),
        mean_change_pp=("change_pp", "mean"),
        mean_patches_before=("patches_before", "mean"),
        mean_patches_after=("patches_after", "mean"),
    )

    total_before = float(results["patches_before"].sum())
    total_after = float(results["patches_after"].sum())
    overall = pd.DataFrame([{
        "images": int(len(results)),
        "locations": int(results["location_id"].nunique()),
        "years": int(results["year"].nunique()),
        "initial_mean_pct": float(results["initial_sand_pct"].mean()),
        "refined_mean_pct": float(results["refined_sand_pct"].mean()),
        "mean_change_pp": float(results["change_pp"].mean()),
        "images_reduced": int((results["change_pp"] < -0.001).sum()),
        "images_increased": int((results["change_pp"] > 0.001).sum()),
        "images_unchanged_within_0_001pp": int((results["change_pp"].abs() <= 0.001).sum()),
        "mean_patches_before": float(results["patches_before"].mean()),
        "mean_patches_after": float(results["patches_after"].mean()),
        "total_patch_reduction_pct": ((total_before - total_after) / total_before * 100.0) if total_before else np.nan,
    }])

    add_check(checks, "Expected method-comparison image count", len(results) == 200, f"Matched images: {len(results)}")
    add_check(checks, "Expected number of locations", results["location_id"].nunique() == 25, f"Locations: {results['location_id'].nunique()}")
    add_check(checks, "Expected year coverage", sorted(results["year"].unique().tolist()) == list(range(2019, 2027)), f"Years: {sorted(results['year'].unique().tolist())}")

    return results, per_image, yearly, overall


# ---------------------------------------------------------------------------
# 4. LOAD AND VALIDATE MEMBER 4 TEMPORAL TABLES
# ---------------------------------------------------------------------------
def validate_change_formulas(df: pd.DataFrame, label: str, checks: list[dict[str, Any]]) -> None:
    numeric = [
        "common_valid_area_km2", "sand_before_km2", "sand_after_km2",
        "gain_km2", "loss_km2", "persistent_sand_km2",
        "persistent_non_sand_km2", "net_change_km2", "total_changed_km2",
        "changed_area_pct", "relative_sand_change_pct",
    ]
    numeric_columns(df, numeric, label)

    formulas = [
        ("Net change = after - before", df["net_change_km2"], df["sand_after_km2"] - df["sand_before_km2"]),
        ("Net change = gain - loss", df["net_change_km2"], df["gain_km2"] - df["loss_km2"]),
        ("Total changed area = gain + loss", df["total_changed_km2"], df["gain_km2"] + df["loss_km2"]),
        ("Before sand = persistent sand + loss", df["sand_before_km2"], df["persistent_sand_km2"] + df["loss_km2"]),
        ("After sand = persistent sand + gain", df["sand_after_km2"], df["persistent_sand_km2"] + df["gain_km2"]),
        (
            "Changed area percentage formula",
            df["changed_area_pct"],
            df["total_changed_km2"] / df["common_valid_area_km2"] * 100.0,
        ),
        (
            "Relative sand change percentage formula",
            df["relative_sand_change_pct"],
            df["net_change_km2"] / df["sand_before_km2"].replace(0, np.nan) * 100.0,
        ),
    ]

    for name, actual, expected in formulas:
        difference = max_abs_difference(actual, expected)
        add_check(
            checks,
            f"{label}: {name}",
            difference <= AREA_TOLERANCE_KM2,
            f"Maximum absolute difference: {difference:.10f}",
        )


def load_temporal_data(checks: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path, label in [
        (ANNUAL_AREA_FILE, "annual_sand_area.csv"),
        (LONG_TERM_FILE, "change_2019_2026.csv"),
        (TEMPORAL_STATS_FILE, "temporal_change_statistics.csv"),
    ]:
        require_file(path, label)

    annual = pd.read_csv(ANNUAL_AREA_FILE)
    long_term = pd.read_csv(LONG_TERM_FILE)
    temporal = pd.read_csv(TEMPORAL_STATS_FILE)

    require_columns(
        annual,
        [
            "location_id", "location", "year", "valid_pixels", "sand_pixels",
            "valid_area_km2", "sand_area_km2", "sand_coverage_pct", "site_row",
        ],
        "annual_sand_area.csv",
    )
    change_columns = [
        "location_id", "site_row", "comparison", "first_year", "last_year",
        "common_valid_area_km2", "sand_before_km2", "sand_after_km2",
        "gain_km2", "loss_km2", "persistent_sand_km2",
        "persistent_non_sand_km2", "net_change_km2", "total_changed_km2",
        "changed_area_pct", "relative_sand_change_pct",
    ]
    require_columns(long_term, change_columns, "change_2019_2026.csv")
    require_columns(temporal, change_columns, "temporal_change_statistics.csv")

    annual = annual.copy()
    long_term = long_term.copy()
    temporal = temporal.copy()
    annual["location_id"] = annual["location_id"].astype(str)
    long_term["location_id"] = long_term["location_id"].astype(str)
    temporal["location_id"] = temporal["location_id"].astype(str)
    annual["year"] = pd.to_numeric(annual["year"], errors="raise").astype(int)
    temporal["first_year"] = pd.to_numeric(temporal["first_year"], errors="raise").astype(int)
    temporal["last_year"] = pd.to_numeric(temporal["last_year"], errors="raise").astype(int)
    long_term["first_year"] = pd.to_numeric(long_term["first_year"], errors="raise").astype(int)
    long_term["last_year"] = pd.to_numeric(long_term["last_year"], errors="raise").astype(int)

    numeric_columns(
        annual,
        ["valid_pixels", "sand_pixels", "valid_area_km2", "sand_area_km2", "sand_coverage_pct"],
        "annual_sand_area.csv",
    )

    # Annual table integrity.
    annual_duplicates = int(annual.duplicated(["location_id", "year"]).sum())
    add_check(checks, "Annual area table has unique location-year records", annual_duplicates == 0, f"Duplicates: {annual_duplicates}")
    add_check(checks, "Annual area table has 200 observations", len(annual) == 200, f"Rows: {len(annual)}")
    add_check(checks, "Annual area table has 25 locations", annual["location_id"].nunique() == 25, f"Locations: {annual['location_id'].nunique()}")
    add_check(checks, "Annual area table covers 2019-2026", sorted(annual["year"].unique().tolist()) == list(range(2019, 2027)), f"Years: {sorted(annual['year'].unique().tolist())}")

    expected_coverage = annual["sand_pixels"] / annual["valid_pixels"] * 100.0
    coverage_error = max_abs_difference(annual["sand_coverage_pct"], expected_coverage)
    add_check(checks, "Annual sand coverage agrees with pixel counts", coverage_error <= 1e-5, f"Maximum difference in percentage points: {coverage_error:.10f}")

    expected_area = annual["sand_pixels"] / annual["valid_pixels"] * annual["valid_area_km2"]
    area_error = max_abs_difference(annual["sand_area_km2"], expected_area)
    add_check(checks, "Annual sand area agrees with pixel counts", area_error <= AREA_TOLERANCE_KM2, f"Maximum area difference: {area_error:.10f} km2")

    # Temporal table counts and formulas.
    add_check(checks, "Temporal statistics has 200 comparisons", len(temporal) == 200, f"Rows: {len(temporal)}")
    comparison_counts = temporal["comparison"].value_counts().to_dict()
    add_check(
        checks,
        "Temporal comparisons include 175 consecutive and 25 full-period records",
        comparison_counts.get("consecutive", 0) == 175 and comparison_counts.get("full_period", 0) == 25,
        f"Comparison counts: {comparison_counts}",
    )
    validate_change_formulas(temporal, "temporal_change_statistics.csv", checks)
    validate_change_formulas(long_term, "change_2019_2026.csv", checks)

    long_period = temporal[temporal["comparison"].astype(str) == "full_period"].copy()
    add_check(checks, "Long-term table has 25 site records", len(long_term) == 25, f"Rows: {len(long_term)}")
    add_check(checks, "Temporal statistics has 25 full-period records", len(long_period) == 25, f"Rows: {len(long_period)}")

    # Long-term table should be the same full-period subset as the 200-row table.
    compare_cols = [
        "sand_before_km2", "sand_after_km2", "gain_km2", "loss_km2",
        "persistent_sand_km2", "persistent_non_sand_km2", "net_change_km2",
        "total_changed_km2", "changed_area_pct", "relative_sand_change_pct",
    ]
    merged_full = long_term.merge(
        long_period,
        on="location_id",
        suffixes=("_long", "_full"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    full_match = (merged_full["_merge"] == "both").all()
    max_full_difference = 0.0
    if full_match:
        errors = [
            max_abs_difference(merged_full[f"{col}_long"], merged_full[f"{col}_full"])
            for col in compare_cols
        ]
        max_full_difference = max(errors) if errors else 0.0
    add_check(
        checks,
        "Long-term table matches full-period rows in temporal statistics",
        bool(full_match) and max_full_difference <= AREA_TOLERANCE_KM2,
        f"All locations matched: {bool(full_match)}; maximum numeric difference: {max_full_difference:.10f}",
    )

    # Verify long-term before/after areas against annual endpoint records.
    annual_2019 = annual[annual["year"] == 2019][["location_id", "sand_area_km2"]].rename(columns={"sand_area_km2": "annual_2019_km2"})
    annual_2026 = annual[annual["year"] == 2026][["location_id", "sand_area_km2"]].rename(columns={"sand_area_km2": "annual_2026_km2"})
    endpoint = long_term.merge(annual_2019, on="location_id", how="left", validate="one_to_one")
    endpoint = endpoint.merge(annual_2026, on="location_id", how="left", validate="one_to_one")
    before_error = max_abs_difference(endpoint["sand_before_km2"], endpoint["annual_2019_km2"])
    after_error = max_abs_difference(endpoint["sand_after_km2"], endpoint["annual_2026_km2"])
    add_check(checks, "2019 long-term areas match annual sand-area table", before_error <= AREA_TOLERANCE_KM2, f"Maximum difference: {before_error:.10f} km2")
    add_check(checks, "2026 long-term areas match annual sand-area table", after_error <= AREA_TOLERANCE_KM2, f"Maximum difference: {after_error:.10f} km2")

    # Annual post-refinement table should agree with Member 3's refined fractions.
    # Fractions in Member 3's summary are rounded to 4 decimals, so a small tolerance is expected.
    return annual, long_term, temporal


# ---------------------------------------------------------------------------
# 5. BUILD TABLES FOR THE REPORT
# ---------------------------------------------------------------------------
def build_output_tables(
    method_results: pd.DataFrame,
    method_yearly: pd.DataFrame,
    method_overall: pd.DataFrame,
    annual: pd.DataFrame,
    long_term: pd.DataFrame,
    temporal: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    annual_yearly = annual.groupby("year", as_index=False).agg(
        image_count=("location_id", "count"),
        locations=("location_id", "nunique"),
        mean_sand_coverage_pct=("sand_coverage_pct", "mean"),
        mean_sand_area_km2=("sand_area_km2", "mean"),
        total_sand_area_km2=("sand_area_km2", "sum"),
    )

    annual_by_type = annual.groupby(["year", "site_row"], as_index=False).agg(
        image_count=("location_id", "count"),
        mean_sand_coverage_pct=("sand_coverage_pct", "mean"),
        mean_sand_area_km2=("sand_area_km2", "mean"),
    )

    consecutive = temporal[temporal["comparison"].astype(str) == "consecutive"].copy()
    temporal_intervals = consecutive.groupby(["first_year", "last_year"], as_index=False).agg(
        site_comparisons=("location_id", "count"),
        mean_gain_km2=("gain_km2", "mean"),
        mean_loss_km2=("loss_km2", "mean"),
        mean_net_change_km2=("net_change_km2", "mean"),
        mean_total_changed_km2=("total_changed_km2", "mean"),
        mean_changed_area_pct=("changed_area_pct", "mean"),
        mean_relative_sand_change_pct=("relative_sand_change_pct", "mean"),
    )
    temporal_intervals["interval"] = (
        temporal_intervals["first_year"].astype(str) + "–" + temporal_intervals["last_year"].astype(str)
    )

    long_term_by_location = long_term.sort_values("net_change_km2").copy()
    long_term_by_type = long_term.groupby("site_row", as_index=False).agg(
        locations=("location_id", "nunique"),
        mean_gain_km2=("gain_km2", "mean"),
        mean_loss_km2=("loss_km2", "mean"),
        mean_net_change_km2=("net_change_km2", "mean"),
        total_gain_km2=("gain_km2", "sum"),
        total_loss_km2=("loss_km2", "sum"),
        total_net_change_km2=("net_change_km2", "sum"),
    )

    # Compare Member 3 refined fractions with Member 4 annual area coverage.
    refined_vs_annual = method_results.merge(
        annual[["location_id", "year", "sand_coverage_pct", "sand_area_km2", "site_row"]],
        on=["location_id", "year"],
        how="left",
        validate="one_to_one",
    )
    refined_vs_annual["refined_vs_annual_difference_pp"] = (
        refined_vs_annual["refined_sand_pct"] - refined_vs_annual["sand_coverage_pct"]
    )

    return {
        "method_per_image": method_results,
        "method_by_year": method_yearly,
        "method_overall": method_overall,
        "annual_by_year": annual_yearly,
        "annual_by_site_type": annual_by_type,
        "temporal_by_interval": temporal_intervals,
        "long_term_by_location": long_term_by_location,
        "long_term_by_site_type": long_term_by_type,
        "refined_vs_annual": refined_vs_annual,
    }


# ---------------------------------------------------------------------------
# 6. VISUALIZATIONS
# ---------------------------------------------------------------------------
def create_method_figures(yearly: pd.DataFrame, method_results: pd.DataFrame) -> None:
    # Figure 1: Initial vs refined coverage.
    plt.figure(figsize=(9, 5))
    plt.plot(yearly["year"], yearly["initial_mean_pct"], marker="o", label="BSI + Otsu")
    plt.plot(yearly["year"], yearly["refined_mean_pct"], marker="o", label="After refinement")
    plt.xlabel("Year")
    plt.ylabel("Predicted Sand Coverage (%)")
    plt.title("Initial vs Refined Sand Coverage")
    plt.xticks(yearly["year"])
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure("Figure_1_Coverage_Comparison.png")

    # Figure 2: Yearly percentage-point difference.
    plt.figure(figsize=(9, 5))
    plt.bar(yearly["year"].astype(str), yearly["mean_change_pp"])
    plt.axhline(0, linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Change in Predicted Sand Coverage (Percentage Points)")
    plt.title("Change in Sand Coverage After Refinement")
    plt.grid(axis="y", alpha=0.3)
    save_figure("Figure_2_Coverage_Difference.png")

    # Figure 3: Connected components; log scale because before/after differ greatly.
    means = [method_results["patches_before"].mean(), method_results["patches_after"].mean()]
    plt.figure(figsize=(7, 5))
    plt.bar(["Before refinement", "After refinement"], means)
    plt.ylabel("Average Number of Connected Components (Log Scale)")
    plt.title("Connected Components Before and After Refinement")
    plt.yscale("log")
    save_figure("Figure_3_Patch_Comparison.png")


def create_temporal_figures(
    annual_by_year: pd.DataFrame,
    annual_by_type: pd.DataFrame,
    temporal_by_interval: pd.DataFrame,
    long_term_by_location: pd.DataFrame,
) -> None:
    # Figure 4: Mean classified sand area per 16 km2 site.
    plt.figure(figsize=(9, 5))
    plt.plot(
        annual_by_year["year"], annual_by_year["mean_sand_area_km2"],
        marker="o", label="Mean sand area per site",
    )
    plt.xlabel("Year")
    plt.ylabel("Mean Classified Sand Area per Site (km²)")
    plt.title("Annual Mean Classified Sand Area")
    plt.xticks(annual_by_year["year"])
    plt.grid(alpha=0.3)
    save_figure("Figure_4_Annual_Sand_Area.png")

    # Figure 5: Coverage trends by site type.
    plt.figure(figsize=(10, 6))
    for site_type, group in annual_by_type.groupby("site_row"):
        plt.plot(
            group["year"], group["mean_sand_coverage_pct"],
            marker="o", label=str(site_type),
        )
    plt.xlabel("Year")
    plt.ylabel("Mean Classified Sand Coverage (%)")
    plt.title("Annual Sand Coverage by Site Type")
    plt.xticks(sorted(annual_by_type["year"].unique()))
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend(title="Site type")
    save_figure("Figure_5_Coverage_by_Site_Type.png")

    # Figure 6: Mean gain and loss per site for consecutive intervals.
    x = np.arange(len(temporal_by_interval))
    width = 0.36
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, temporal_by_interval["mean_gain_km2"], width, label="Mean sand gain")
    plt.bar(x + width / 2, temporal_by_interval["mean_loss_km2"], width, label="Mean sand loss")
    plt.xticks(x, temporal_by_interval["interval"], rotation=30, ha="right")
    plt.xlabel("Comparison interval")
    plt.ylabel("Mean Area per Site (km²)")
    plt.title("Mean Sand Gain and Loss by Consecutive Interval")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    save_figure("Figure_6_Gain_and_Loss_by_Interval.png")

    # Figure 7: Net classified-sand change per site, 2019-2026.
    ordered = long_term_by_location.sort_values("net_change_km2")
    plt.figure(figsize=(10, 8))
    plt.barh(ordered["location_id"], ordered["net_change_km2"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Net Change in Classified Sand Area (km²)")
    plt.ylabel("Location")
    plt.title("Net Classified Sand-Coverage Change by Location (2019–2026)")
    plt.grid(axis="x", alpha=0.3)
    save_figure("Figure_7_Net_Change_by_Location.png")

    # Figure 8: Mean changed area percentage by consecutive interval.
    plt.figure(figsize=(10, 5))
    plt.bar(temporal_by_interval["interval"], temporal_by_interval["mean_changed_area_pct"])
    plt.xlabel("Comparison interval")
    plt.ylabel("Mean Changed Area (%)")
    plt.title("Mean Percentage of Area Reclassified by Interval")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_figure("Figure_8_Changed_Area_by_Interval.png")


def create_selected_mask_figure() -> bool:
    """Compare existing BSI masks with refined masks for representative sites."""
    if rasterio is None:
        print("[SKIP] Mask comparison figure: rasterio is not installed.")
        return False

    examples = [
        ("L1_Abqaiq_N", 2019),
        ("L10_DuneNE_km11_9", 2026),
        ("L16_SandSW_km36_4", 2024),
    ]
    available: list[tuple[str, int, np.ndarray, np.ndarray]] = []

    for location, year in examples:
        bsi_path = BSI_MASK_DIR / f"{location}_{year}_BSI_Otsu_mask.tif"
        refined_path = REFINED_MASK_DIR / str(year) / f"{location}_{year}_refined.tif"
        if not bsi_path.exists() or not refined_path.exists():
            print(f"[SKIP] Missing mask pair for {location}, {year}.")
            continue

        with rasterio.open(bsi_path) as src:
            initial = src.read(1)
            bsi_shape = src.shape
            bsi_transform = src.transform
            bsi_crs = src.crs
        with rasterio.open(refined_path) as src:
            refined = src.read(1)
            refined_shape = src.shape
            refined_transform = src.transform
            refined_crs = src.crs

        if (
            bsi_shape != refined_shape
            or bsi_transform != refined_transform
            or bsi_crs != refined_crs
        ):
            print(f"[SKIP] Grid mismatch for {location}, {year}; not plotting misaligned masks.")
            continue

        available.append((location, year, initial, refined))

    if not available:
        print("[SKIP] No matching initial/refined mask pairs found for the selected examples.")
        return False

    fig, axes = plt.subplots(len(available), 2, figsize=(9, 4 * len(available)), squeeze=False)
    for row_index, (location, year, initial, refined) in enumerate(available):
        initial_display = np.ma.masked_where(initial == 255, initial)
        refined_display = np.ma.masked_where(refined == 255, refined)
        axes[row_index, 0].imshow(initial_display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 0].set_title(f"{location} — {year}\nBSI + Otsu")
        axes[row_index, 1].imshow(refined_display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 1].set_title(f"{location} — {year}\nAfter refinement")
        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle("Selected Initial and Refined Segmentation Masks", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_9_Selected_Mask_Comparisons.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def copy_selected_previews() -> int:
    """Copy existing five-panel previews to the evaluation output folder."""
    if not PREVIEW_DIR.exists():
        return 0

    candidates = [
        "L1_Abqaiq_N_2019.png",
        "L10_DuneNE_km11_9_2026.png",
        "L16_SandSW_km36_4_2024.png",
    ]
    SELECTED_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for filename in candidates:
        source = PREVIEW_DIR / filename
        if source.exists():
            shutil.copy2(source, SELECTED_PREVIEW_DIR / filename)
            copied += 1
    return copied


# ---------------------------------------------------------------------------
# 7. AUTOMATIC PSEUDO-LABEL AGREEMENT EVALUATION
# ---------------------------------------------------------------------------
def _read_binary_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a 0/1/255 mask and return the array plus raster metadata."""
    if rasterio is None:
        raise RuntimeError("rasterio is required to compare segmentation masks.")
    with rasterio.open(path) as src:
        mask = src.read(1)
        metadata = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
            "profile": src.profile.copy(),
            "dataset_valid": src.read_masks(1) > 0,
        }
    unexpected = np.setdiff1d(np.unique(mask), np.array([0, 1, 255]))
    if unexpected.size:
        raise ValueError(
            f"Mask {path} contains unexpected values {unexpected.tolist()}; "
            "expected only 0 (non-sand), 1 (sand), and 255 (invalid)."
        )
    return mask, metadata


def _agreement_counts(
    pseudo_label: np.ndarray,
    refined_prediction: np.ndarray,
    pseudo_valid: np.ndarray,
    refined_valid: np.ndarray,
) -> dict[str, int | float]:
    """Treat BSI+Otsu as reference and the refined mask as prediction."""
    valid = (
        pseudo_valid
        & refined_valid
        & (pseudo_label != 255)
        & (refined_prediction != 255)
    )
    reference_sand = pseudo_label == 1
    predicted_sand = refined_prediction == 1

    tp = int((predicted_sand & reference_sand & valid).sum())
    fp = int((predicted_sand & ~reference_sand & valid).sum())
    fn = int((~predicted_sand & reference_sand & valid).sum())
    tn = int((~predicted_sand & ~reference_sand & valid).sum())
    valid_pixels = int(valid.sum())

    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "valid_pixels": valid_pixels,
        "IoU": tp / union if union else np.nan,
        "Dice": 2 * tp / dice_denominator if dice_denominator else np.nan,
        "Precision": tp / precision_denominator if precision_denominator else np.nan,
        "Recall": tp / recall_denominator if recall_denominator else np.nan,
        "agreement_pct": 100.0 * (tp + tn) / valid_pixels if valid_pixels else np.nan,
    }


def _write_pseudo_difference_mask(
    pseudo_label: np.ndarray,
    refined_prediction: np.ndarray,
    valid: np.ndarray,
    reference_profile: dict[str, Any],
    output_path: Path,
) -> None:
    """Save class transitions: 0 both non-sand, 1 both sand, 2 removed, 3 added, 255 invalid."""
    if rasterio is None:
        return
    difference = np.full(pseudo_label.shape, 255, dtype=np.uint8)
    reference_sand = pseudo_label == 1
    predicted_sand = refined_prediction == 1
    difference[valid & ~reference_sand & ~predicted_sand] = 0
    difference[valid & reference_sand & predicted_sand] = 1
    difference[valid & reference_sand & ~predicted_sand] = 2
    difference[valid & ~reference_sand & predicted_sand] = 3

    profile = reference_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(difference, 1)


def _summarize_agreement(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return per-image macro means and pooled (micro) metrics overall and by year."""
    source = pd.DataFrame(records)
    metric_names = ["IoU", "Dice", "Precision", "Recall", "agreement_pct"]
    rows: list[dict[str, Any]] = []

    groups: list[tuple[str, Any, pd.DataFrame]] = [("Overall", "All", source)]
    groups.extend(("Year", int(year), frame) for year, frame in source.groupby("year"))

    for group_type, group_name, frame in groups:
        macro = {
            metric: float(frame[metric].mean()) if frame[metric].notna().any() else np.nan
            for metric in metric_names
        }
        rows.append({
            "group_type": group_type,
            "group": group_name,
            "aggregation": "macro_mean_per_image",
            "images": int(len(frame)),
            "valid_pixels": int(frame["valid_pixels"].sum()),
            **macro,
        })

        tp = int(frame["TP"].sum())
        fp = int(frame["FP"].sum())
        fn = int(frame["FN"].sum())
        tn = int(frame["TN"].sum())
        union = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        rows.append({
            "group_type": group_type,
            "group": group_name,
            "aggregation": "micro_pooled_pixels",
            "images": int(len(frame)),
            "valid_pixels": int(frame["valid_pixels"].sum()),
            "IoU": tp / union if union else np.nan,
            "Dice": 2 * tp / dice_denominator if dice_denominator else np.nan,
            "Precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "Recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "agreement_pct": 100.0 * (tp + tn) / (tp + tn + fp + fn)
            if (tp + tn + fp + fn) else np.nan,
        })
    return pd.DataFrame(rows)


def try_pseudo_label_evaluation(
    checks: list[dict[str, Any]], method_results: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compare all available refined masks against BSI+Otsu automatic pseudo-labels.

    BSI+Otsu is the reference (pseudo-label); Texture+Morphology is the prediction.
    Scores represent agreement between two project methods, not true accuracy.
    Both masks must share the same raster grid and use 0/1/255 encoding.
    """
    check_name = "Automatic pseudo-label agreement metrics"
    if rasterio is None:
        add_check(
            checks, check_name, False,
            "NOT RUN: rasterio is not installed; mask rasters cannot be compared.",
            status_override="NOT AVAILABLE",
        )
        return None, None

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []

    for row in method_results.itertuples(index=False):
        location = str(row.location)
        year = int(row.year)
        pair_id = f"{location}_{year}"
        pseudo_path = BSI_MASK_DIR / f"{location}_{year}_BSI_Otsu_mask.tif"
        refined_path = REFINED_MASK_DIR / str(year) / f"{location}_{year}_refined.tif"

        if not pseudo_path.exists() or not refined_path.exists():
            missing.append(pair_id)
            continue

        try:
            pseudo, pseudo_meta = _read_binary_mask(pseudo_path)
            refined, refined_meta = _read_binary_mask(refined_path)
        except Exception as exc:
            invalid.append(f"{pair_id}: {exc}")
            continue

        same_grid = (
            pseudo_meta["shape"] == refined_meta["shape"]
            and pseudo_meta["transform"] == refined_meta["transform"]
            and pseudo_meta["crs"] == refined_meta["crs"]
        )
        if not same_grid:
            invalid.append(f"{pair_id}: raster grid mismatch")
            continue

        pseudo_valid = pseudo_meta["dataset_valid"]
        refined_valid = refined_meta["dataset_valid"]
        valid = (
            pseudo_valid & refined_valid & (pseudo != 255) & (refined != 255)
        )
        counts = _agreement_counts(
            pseudo, refined, pseudo_valid, refined_valid
        )
        records.append({
            "location": location,
            "year": year,
            "pseudo_label_source": "BSI + Otsu",
            "compared_method": "Texture + Morphology",
            **counts,
        })

        difference_path = (
            PSEUDO_DIFFERENCE_DIR / str(year)
            / f"{location}_{year}_pseudo_label_difference.tif"
        )
        _write_pseudo_difference_mask(
            pseudo, refined, valid, pseudo_meta["profile"], difference_path
        )

    if records:
        per_image = pd.DataFrame(records)
        summary = _summarize_agreement(records)
        per_image.to_csv(OUTPUT_DIR / "pseudo_label_agreement_per_image.csv", index=False)
        summary.to_csv(OUTPUT_DIR / "pseudo_label_agreement_summary.csv", index=False)

        complete = len(records) == len(method_results) and not missing and not invalid
        add_check(
            checks, check_name, complete,
            f"Compared: {len(records)}/{len(method_results)} images; "
            f"missing pairs: {len(missing)}; invalid/grid-mismatched pairs: {len(invalid)}. "
            "Metrics measure agreement with BSI+Otsu pseudo-labels, not independent accuracy.",
            status_override=None if complete else "CHECK",
        )
        if missing:
            pd.DataFrame({"missing_pair": missing}).to_csv(
                OUTPUT_DIR / "pseudo_label_missing_pairs.csv", index=False
            )
        if invalid:
            pd.DataFrame({"invalid_pair": invalid}).to_csv(
                OUTPUT_DIR / "pseudo_label_invalid_pairs.csv", index=False
            )
        print(
            f"[INFO] Pseudo-label agreement: {len(records)}/{len(method_results)} image pairs; "
            f"{len(missing)} missing; {len(invalid)} invalid."
        )
        return per_image, summary

    add_check(
        checks, check_name, False,
        "NOT RUN: no compatible BSI+Otsu and refined mask pairs were found.",
        status_override="NOT AVAILABLE",
    )
    if missing:
        print("[INFO] Pseudo-label mask pairs missing:", ", ".join(missing[:10]))
    if invalid:
        print("[INFO] Pseudo-label mask pairs invalid:", "; ".join(invalid[:5]))
    return None, None


def create_pseudo_label_difference_figure() -> bool:
    """Create 3-panel examples of pseudo-label, refined result, and changed pixels."""
    if rasterio is None:
        return False

    examples = [
        ("L1_Abqaiq_N", 2019),
        ("L10_DuneNE_km11_9", 2026),
        ("L16_SandSW_km36_4", 2024),
    ]
    available: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []

    for location, year in examples:
        pseudo_path = BSI_MASK_DIR / f"{location}_{year}_BSI_Otsu_mask.tif"
        refined_path = REFINED_MASK_DIR / str(year) / f"{location}_{year}_refined.tif"
        if not pseudo_path.exists() or not refined_path.exists():
            continue
        try:
            pseudo, pseudo_meta = _read_binary_mask(pseudo_path)
            refined, refined_meta = _read_binary_mask(refined_path)
        except Exception:
            continue
        if not (
            pseudo_meta["shape"] == refined_meta["shape"]
            and pseudo_meta["transform"] == refined_meta["transform"]
            and pseudo_meta["crs"] == refined_meta["crs"]
        ):
            continue
        valid = (
            pseudo_meta["dataset_valid"] & refined_meta["dataset_valid"]
            & (pseudo != 255) & (refined != 255)
        )
        difference = np.full(pseudo.shape, 255, dtype=np.uint8)
        ref_sand = pseudo == 1
        pred_sand = refined == 1
        difference[valid & ~ref_sand & ~pred_sand] = 0
        difference[valid & ref_sand & pred_sand] = 1
        difference[valid & ref_sand & ~pred_sand] = 2
        difference[valid & ~ref_sand & pred_sand] = 3
        available.append((location, year, pseudo, refined, difference))

    if not available:
        print("[SKIP] No compatible pseudo-label example pairs for the difference figure.")
        return False

    fig, axes = plt.subplots(len(available), 3, figsize=(13, 4 * len(available)), squeeze=False)
    for idx, (location, year, pseudo, refined, difference) in enumerate(available):
        axes[idx, 0].imshow(np.ma.masked_where(pseudo == 255, pseudo), cmap="gray", vmin=0, vmax=1)
        axes[idx, 0].set_title(f"{location} — {year}\nPseudo-Label: BSI + Otsu")
        axes[idx, 1].imshow(np.ma.masked_where(refined == 255, refined), cmap="gray", vmin=0, vmax=1)
        axes[idx, 1].set_title("Texture + Morphology")
        axes[idx, 2].imshow(np.ma.masked_where(difference == 255, difference), vmin=0, vmax=3, interpolation="nearest")
        axes[idx, 2].set_title("Agreement / Changed Classes\n0 both non-sand | 1 both sand | 2 removed | 3 added")
        for ax in axes[idx]:
            ax.axis("off")

    fig.suptitle("Automatic Pseudo-Label Agreement Examples", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "Figure_10_Pseudo_Label_Agreement_Examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    print("\n========== MEMBER 5 EVALUATION PIPELINE ==========")
    print("Project root:", ROOT)

    # Method comparison: Member 2 vs Member 3.
    method_results, method_per_image, method_yearly, method_overall = load_method_comparison(checks)

    # Member 4 temporal outputs.
    annual, long_term, temporal = load_temporal_data(checks)

    # Annual table should represent the refined masks from Member 3.
    annual_refined = method_results.merge(
        annual[["location_id", "year", "sand_coverage_pct", "sand_area_km2"]],
        on=["location_id", "year"],
        how="left",
        validate="one_to_one",
    )
    annual_refinement_error = max_abs_difference(
        annual_refined["refined_sand_pct"], annual_refined["sand_coverage_pct"]
    )
    add_check(
        checks,
        "Annual sand coverage agrees with refined segmentation summary",
        annual_refinement_error <= PERCENTAGE_TOLERANCE,
        f"Maximum difference: {annual_refinement_error:.6f} percentage points (tolerance {PERCENTAGE_TOLERANCE})",
    )

    tables = build_output_tables(method_per_image, method_yearly, method_overall, annual, long_term, temporal)

    # Save data tables.
    save_csv(tables["method_per_image"], "method_comparison_per_image.csv")
    save_csv(tables["method_by_year"], "method_comparison_by_year.csv")
    save_csv(tables["method_overall"], "method_comparison_overall.csv")
    save_csv(tables["annual_by_year"], "annual_sand_area_by_year.csv")
    save_csv(tables["annual_by_site_type"], "annual_sand_area_by_site_type.csv")
    save_csv(tables["temporal_by_interval"], "temporal_change_by_interval.csv")
    save_csv(tables["long_term_by_location"], "long_term_change_by_location.csv")
    save_csv(tables["long_term_by_site_type"], "long_term_change_by_site_type.csv")
    save_csv(tables["refined_vs_annual"], "refined_vs_annual_area_check.csv")

    # Create figures.
    create_method_figures(tables["method_by_year"], method_results)
    create_temporal_figures(
        tables["annual_by_year"],
        tables["annual_by_site_type"],
        tables["temporal_by_interval"],
        tables["long_term_by_location"],
    )
    mask_figure_created = create_selected_mask_figure()
    copied_previews = copy_selected_previews()

    # Compare refined masks against BSI+Otsu automatic pseudo-labels.
    pseudo_per_image, pseudo_summary = try_pseudo_label_evaluation(checks, method_results)
    pseudo_figure_created = create_pseudo_label_difference_figure()

    # Provenance file checks.
    if PROVENANCE_FILE.exists():
        with PROVENANCE_FILE.open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        expected_mask_values = {"0": "non-sand", "1": "sand", "255": "invalid"}
        mask_values_ok = provenance.get("mask_values") == expected_mask_values
        add_check(checks, "Provenance mask values are documented as 0/1/255", mask_values_ok, str(provenance.get("mask_values")))
        add_check(
            checks,
            "Provenance observation/comparison counts are 200",
            provenance.get("observations") == 200 and provenance.get("comparisons") == 200,
            f"observations={provenance.get('observations')}; comparisons={provenance.get('comparisons')}",
        )
    else:
        add_check(checks, "Temporal provenance file available", False, f"Not found: {PROVENANCE_FILE}", status_override="NOT AVAILABLE")

    # Save validation report and plain-language summary.
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUTPUT_DIR / "validation_report.csv", index=False)

    overall_row = method_overall.iloc[0]
    consecutive = temporal[temporal["comparison"].astype(str) == "consecutive"]
    full_period = long_term

    pseudo_metric_lines: list[str] = []
    if pseudo_summary is not None:
        pseudo_macro = pseudo_summary[
            (pseudo_summary["group_type"] == "Overall")
            & (pseudo_summary["aggregation"] == "macro_mean_per_image")
        ].iloc[0]
        pseudo_micro = pseudo_summary[
            (pseudo_summary["group_type"] == "Overall")
            & (pseudo_summary["aggregation"] == "micro_pooled_pixels")
        ].iloc[0]
        pseudo_metric_lines = [
            "Overall macro mean per image:",
            *[f"  {metric}: {pseudo_macro[metric]:.4f}" for metric in ["IoU", "Dice", "Precision", "Recall", "agreement_pct"]],
            "Overall micro score across pooled pixels:",
            *[f"  {metric}: {pseudo_micro[metric]:.4f}" for metric in ["IoU", "Dice", "Precision", "Recall", "agreement_pct"]],
        ]
    else:
        pseudo_metric_lines = ["Pseudo-label agreement metrics: NOT AVAILABLE (compatible mask pairs were not found)."]

    summary_lines = [
        "MEMBER 5 EVALUATION SUMMARY",
        "===========================",
        f"Method-comparison images: {int(overall_row['images'])}",
        f"Locations: {int(overall_row['locations'])}",
        f"Years: {int(overall_row['years'])}",
        f"Initial mean predicted sand coverage: {overall_row['initial_mean_pct']:.3f}%",
        f"Refined mean predicted sand coverage: {overall_row['refined_mean_pct']:.3f}%",
        f"Mean coverage difference: {overall_row['mean_change_pp']:.3f} percentage points",
        f"Images with reduced coverage: {int(overall_row['images_reduced'])}",
        f"Images with increased coverage: {int(overall_row['images_increased'])}",
        f"Mean patches before: {overall_row['mean_patches_before']:.2f}",
        f"Mean patches after: {overall_row['mean_patches_after']:.2f}",
        f"Total patch-count reduction: {overall_row['total_patch_reduction_pct']:.3f}%",
        "",
        "TEMPORAL RESULTS (2019-2026; sums across 25 site records)",
        f"Total sand gain: {full_period['gain_km2'].sum():.4f} km2",
        f"Total sand loss: {full_period['loss_km2'].sum():.4f} km2",
        f"Total net classified-sand change: {full_period['net_change_km2'].sum():.4f} km2",
        f"Sites with net decrease: {(full_period['net_change_km2'] < 0).sum()}",
        f"Sites with net increase: {(full_period['net_change_km2'] > 0).sum()}",
        f"Consecutive temporal comparisons: {len(consecutive)}",
        "",
        "PSEUDO-LABEL AGREEMENT EVALUATION",
        "BSI + Otsu masks are used as automatic pseudo-labels (weak reference masks).",
        "Texture + Morphology masks are treated as predictions and compared with those pseudo-labels.",
        "IoU, Dice, Precision and Recall measure agreement with the BSI + Otsu baseline only.",
        "They are NOT independent segmentation-accuracy scores because manually annotated Ground Truth is unavailable.",
        f"Pseudo-label image pairs evaluated: {len(pseudo_per_image) if pseudo_per_image is not None else 0}",
        *pseudo_metric_lines,
        f"Pseudo-label difference figure created: {pseudo_figure_created}",
        "Difference-mask values: 0=both non-sand, 1=both sand, 2=initial sand removed, 3=sand added by refinement, 255=invalid.",
        "",
        "TEMPORAL INTERPRETATION NOTE",
        "Temporal measurements describe changes in areas classified as sand by the image-processing workflow.",
        "They do not independently validate physical dune movement.",
        "",
        f"Mask comparison figure created: {mask_figure_created}",
        f"Existing preview panels copied: {copied_previews}",
        f"Validation checks: {(checks_df['status'] == 'PASS').sum()} PASS; {(checks_df['status'] == 'CHECK').sum()} CHECK; {(checks_df['status'] == 'NOT AVAILABLE').sum()} NOT AVAILABLE",
    ]
    (OUTPUT_DIR / "evaluation_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    pseudo_notes = [
        "AUTOMATIC PSEUDO-LABEL EVALUATION NOTES",
        "========================================",
        "Reference / pseudo-label: BSI + Otsu binary segmentation mask.",
        "Compared prediction: Texture + Morphology refined binary segmentation mask.",
        "Positive class: sand (pixel value 1). Non-sand: 0. Invalid/ignored: 255.",
        "Both masks must have identical dimensions, transform, and CRS.",
        "IoU, Dice, Precision, and Recall describe agreement with the BSI + Otsu pseudo-labels only.",
        "They must NOT be described as true segmentation accuracy or independent validation.",
        "Precision = refined sand pixels overlapping BSI+Otsu sand / all refined sand pixels.",
        "Recall = refined sand pixels overlapping BSI+Otsu sand / all BSI+Otsu sand pixels.",
        "The evaluation does not use manually annotated Ground Truth masks because none are available.",
        "Difference mask classes: 0=both non-sand; 1=both sand; 2=BSI+Otsu sand removed by refinement; 3=sand added by refinement; 255=invalid.",
        "Macro summary = mean of per-image scores. Micro summary = metrics calculated from pooled pixel counts.",
    ]
    (OUTPUT_DIR / "pseudo_label_evaluation_notes.txt").write_text("\n".join(pseudo_notes), encoding="utf-8")

    # Console summary.
    print("\n========== OVERALL METHOD COMPARISON ==========")
    print(method_overall.round(3).to_string(index=False))
    print("\n========== ANNUAL TEMPORAL SUMMARY ==========")
    print(tables["annual_by_year"].round(3).to_string(index=False))
    print("\n========== LONG-TERM CHANGE SUMMARY ==========")
    print(
        full_period[["gain_km2", "loss_km2", "net_change_km2", "total_changed_km2"]]
        .sum().round(4).to_string()
    )
    print("\n========== VALIDATION REPORT ==========")
    print(checks_df.to_string(index=False))
    print("\nAll available evaluation outputs saved to:", OUTPUT_DIR)
    print("Pseudo-label metrics are agreement measures only; no independent Ground Truth accuracy is claimed.")


if __name__ == "__main__":
    main()
