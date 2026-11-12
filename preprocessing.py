"""ARTI 403: validate Sentinel-2 chips and export masked surface reflectance.

Run ``python preprocessing.py --help``. Source data are opened read-only.
No segmentation, registration, radiometric enhancement or change detection is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter


BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
# Conservative SCL policy: saturated/defective and snow/ice are also unsuitable.
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
PIPELINE = "arti403-preprocessing-v1"
DISPLAY_RANGE = (0.0, 0.6)  # Fixed for every image, year and location; display only.


def sha256(path: Path) -> str:
    """Hash a file without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(root: Path, relative: str) -> Path:
    """Resolve metadata paths, refusing absolute paths and directory traversal."""
    relative = Path(relative)
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Metadata path is outside source directory: {relative}")
    return resolved


def grid_signature(ds) -> tuple:
    return (ds.width, ds.height, ds.crs, ds.transform, ds.bounds)


def prepare_arrays(raw: np.ndarray, raster_valid: np.ndarray):
    """Return unmasked Float32 reflectance, categorical SCL, validity and analysis.

    The input and unmasked reflectance remain unchanged. Classes 2 (dark area),
    4 (vegetation), 5 (bare soil), 6 (water), and 7 (unclassified) remain valid.
    Bright sand can be misclassified by SCL; this mask is not ground truth.
    """
    if raw.ndim != 3 or raw.shape[0] != 7:
        raise ValueError("Expected seven bands in band-first order")
    if raster_valid.shape != raw.shape[1:]:
        raise ValueError("Source mask and image dimensions differ")
    scl = raw[6].astype(np.uint8, copy=True)
    if np.any(raw[6] > 11):
        raise ValueError("Unexpected SCL class outside 0–11")
    reflectance = raw[:6].astype(np.float32) / np.float32(10000)
    valid = np.asarray(raster_valid, dtype=bool) & ~np.isin(scl, INVALID_SCL)
    valid &= np.all(np.isfinite(reflectance), axis=0)
    analysis = reflectance.copy()
    analysis[:, ~valid] = np.nan
    return reflectance, scl, valid, analysis


def smooth_for_display(image: np.ndarray, method="gaussian", strength=1.0):
    """Smooth a COPY, per band, without filling invalid pixels or mixing bands.

    Gaussian uses normalized finite-pixel weights; median ignores NaNs. Smoothing
    can remove small dune features and alter reflectance. Never used for exports.
    """
    if method not in {"gaussian", "median"} or strength <= 0:
        raise ValueError("Choose gaussian/median and a positive strength")
    result = np.full_like(image, np.nan, dtype=np.float32)
    for index, band in enumerate(image):
        finite = np.isfinite(band)
        if method == "gaussian":
            weight = gaussian_filter(finite.astype(float), strength)
            numerator = gaussian_filter(np.where(finite, band, 0.0), strength)
            np.divide(numerator, weight, out=result[index], where=weight > 0)
        else:
            # scipy generic_filter supports a NaN-aware local median.
            from scipy.ndimage import generic_filter

            size = max(1, int(strength))
            if size % 2 == 0:
                size += 1
            def finite_median(values):
                values = values[np.isfinite(values)]
                return np.median(values) if values.size else np.nan
            result[index] = generic_filter(band, finite_median, size=size)
        result[index, ~finite] = np.nan
    return result


def inspect_sources(root: Path):
    """Preflight every metadata record and inventory; flag full location groups
    when temporal grids differ. Read all pixels to detect corrupt rasters.
    """
    root = root.resolve()
    table = pd.read_csv(root / "metadata.csv", keep_default_na=False)
    required = {"location_id", "location_name", "year", "acquisition_date",
                "geotiff_path", "png_path", "geotiff_sha256", "width_px",
                "height_px", "resolution_m", "crs", "bbox_minx", "bbox_miny",
                "bbox_maxx", "bbox_maxy"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(table.columns))}")
    records = table.to_dict("records")
    issues, signatures, observed = [], defaultdict(list), set()
    duplicate_keys = table.duplicated(["location_id", "year"], keep=False)
    duplicate_paths = table.duplicated("geotiff_path", keep=False)
    duplicate_hashes = table.duplicated("geotiff_sha256", keep=False)
    for index, row in enumerate(records):
        row["errors"], row["warnings"] = [], []
        if duplicate_keys.iloc[index] or duplicate_paths.iloc[index]:
            row["errors"].append("Duplicate location/year or source path in metadata")
        if duplicate_hashes.iloc[index]:
            row["errors"].append("Duplicate source checksum in metadata")
        try:
            path = source_path(root, row["geotiff_path"])
            observed.add(path)
            row["source_sha256"] = sha256(path)
            if row["source_sha256"] != row["geotiff_sha256"]:
                raise ValueError("GeoTIFF SHA-256 differs from authoritative metadata")
            expected_stem = f"{row['location_name']}_{row['year']}"
            if path.stem != expected_stem or path.parent.name != str(row["year"]):
                raise ValueError("Source filename/year does not match metadata")
            if pd.Timestamp(row["acquisition_date"]).year != int(row["year"]):
                raise ValueError("Acquisition date does not match year")
            with rasterio.open(path) as ds:
                if ds.count != 7 or ds.dtypes != ("uint16",) * 7:
                    raise ValueError("Expected seven UInt16 bands")
                if (ds.width, ds.height) != (400, 400) or (ds.width, ds.height) != (int(row["width_px"]), int(row["height_px"])):
                    raise ValueError("Image dimensions differ from 400×400 or metadata")
                if ds.crs != rasterio.crs.CRS.from_string(row["crs"]) or ds.crs.to_epsg() != 32639:
                    raise ValueError("CRS differs from metadata or EPSG:32639")
                if ds.res != (10.0, 10.0) or ds.res != (float(row["resolution_m"]),) * 2:
                    raise ValueError("Pixel resolution differs from 10 m or metadata")
                bounds = tuple(float(row[key]) for key in ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"))
                if tuple(ds.bounds) != bounds or ds.transform != from_bounds(*bounds, ds.width, ds.height):
                    raise ValueError("Transform/bounds differ from authoritative metadata")
                for band, expected in zip(ds.descriptions, (*BANDS, "SCL")):
                    if not band or not band.startswith(expected):
                        raise ValueError(f"Missing/mismatched spectral band description: {band}")
                tags = ds.tags()
                if tags.get("location") != row["location_name"] or tags.get("acquisition_date") != row["acquisition_date"]:
                    raise ValueError("Embedded location/acquisition date differs from metadata")
                raw = ds.read()
                prepare_arrays(raw, np.all(ds.read_masks() > 0, axis=0))
                signatures[row["location_id"]].append((index, grid_signature(ds)))
        except (OSError, ValueError, rasterio.errors.RasterioError) as error:
            row["errors"].append(str(error))
        try:
            png = source_path(root, row["png_path"])
            if png.stem != f"{row['location_name']}_{row['year']}":
                raise ValueError("PNG filename differs from metadata")
            original = plt.imread(png)
            if original.shape != (400, 400, 3):
                raise ValueError("PNG dimensions/channels differ from 400×400 RGB")
        except (OSError, ValueError) as error:
            row["warnings"].append(f"Visualization PNG: {error}")
    for location, members in signatures.items():
        if len({signature for _, signature in members}) > 1:
            for index, _ in members:
                records[index]["errors"].append(f"Temporal grid mismatch at {location}; no warp performed")
    actual = {p.resolve() for p in (root / "geotiff").rglob("*") if p.suffix.lower() in {".tif", ".tiff"}}
    for path in sorted(actual - observed):
        issues.append({"severity": "error", "file": str(path.relative_to(root)), "issue": "GeoTIFF has no metadata record; not processed"})
    for location, group in table.groupby("location_id"):
        if set(group["year"].astype(int)) != set(range(2019, 2027)):
            issues.append({"severity": "error", "file": str(location), "issue": "Incomplete/extra temporal coverage; expected 2019–2026"})
    if len(table) != 200 or set(table["location_id"]) != {f"L{i}" for i in range(1, 26)}:
        issues.append({"severity": "error", "file": "metadata.csv", "issue": "Expected 200 records for L1–L25"})
    for row in records:
        for level, key in (("error", "errors"), ("warning", "warnings")):
            issues.extend({"severity": level, "file": row["geotiff_path"], "issue": item} for item in row[key])
    return records, issues


def load_observation(path: Path):
    """Load scientific arrays and export metadata; do not modify source pixels."""
    with rasterio.open(path) as ds:
        arrays = prepare_arrays(ds.read(), np.all(ds.read_masks() > 0, axis=0))
        return (*arrays, ds.profile.copy(), ds.descriptions, ds.tags())


def output_paths(output: Path, row: dict):
    stem, year = Path(row["geotiff_path"]).stem, str(row["year"])
    return (output / "geotiff" / year / f"{stem}_processed.tif",
            output / "masks" / year / f"{stem}_valid.tif",
            output / "masks" / year / f"{stem}_scl.tif")


def verify_outputs(paths, analysis, valid, scl, profile, descriptions, fingerprint):
    """Reopen every export and verify numerical values and georeferencing."""
    expected_arrays = (analysis, valid.astype(np.uint8)[None], scl[None])
    for path, expected in zip(paths, expected_arrays):
        with rasterio.open(path) as ds:
            if (ds.crs, ds.transform, ds.width, ds.height) != (profile["crs"], profile["transform"], profile["width"], profile["height"]):
                raise ValueError(f"Output georeferencing mismatch: {path}")
            if ds.tags().get("fingerprint") != fingerprint:
                raise ValueError(f"Output provenance mismatch: {path}")
            if ds.count != expected.shape[0] or ds.dtypes != (str(expected.dtype),) * expected.shape[0]:
                raise ValueError(f"Output count/type mismatch: {path}")
            if not np.array_equal(ds.read(), expected, equal_nan=True):
                raise ValueError(f"Output numerical mismatch: {path}")
            if path == paths[0] and (ds.descriptions != tuple(descriptions[:6]) or not np.isnan(ds.nodata)):
                raise ValueError(f"Output descriptions/NaN nodata mismatch: {path}")


def export_observation(root: Path, output: Path, row: dict):
    """Export or reuse verified pipeline-owned files, with content-based caching."""
    reflectance, scl, valid, analysis, profile, descriptions, tags = load_observation(source_path(root, row["geotiff_path"]))
    paths = output_paths(output, row)
    fingerprint = hashlib.sha256((row["source_sha256"] + sha256(Path(__file__)) + PIPELINE).encode()).hexdigest()
    cached = False
    # Never overwrite files created by a teammate or a different tool.
    for path in paths:
        if path.exists():
            with rasterio.open(path) as ds:
                if ds.tags().get("pipeline") != PIPELINE:
                    raise ValueError(f"Refusing to overwrite non-pipeline file: {path}")
    if all(path.exists() for path in paths):
        try:
            verify_outputs(paths, analysis, valid, scl, profile, descriptions, fingerprint)
            cached = True
        except (ValueError, rasterio.errors.RasterioError):
            pass
    if not cached:
        for path, array, names in zip(paths, (analysis, valid.astype(np.uint8)[None], scl[None]),
                                       (descriptions[:6], ("validity_1_valid_0_invalid",), (descriptions[6],))):
            path.parent.mkdir(parents=True, exist_ok=True)
            settings = profile.copy()
            settings.update(count=array.shape[0], dtype=str(array.dtype), nodata=np.nan if path == paths[0] else None,
                            compress="deflate", predictor=3 if path == paths[0] else 2)
            temporary = path.with_suffix(".tmp.tif")
            try:
                with rasterio.open(temporary, "w", **settings) as ds:
                    ds.write(array)
                    ds.descriptions = tuple(names)
                    ds.update_tags(**{**tags, "pipeline": PIPELINE, "fingerprint": fingerprint,
                                      "source_sha256": row["source_sha256"], "units": "surface reflectance" if path == paths[0] else "categorical",
                                      "scale": "already scaled; multiplier 1" if path == paths[0] else "categorical; no scaling",
                                      "invalid_scl": ",".join(map(str, INVALID_SCL)), "filtering": "disabled"})
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        verify_outputs(paths, analysis, valid, scl, profile, descriptions, fingerprint)
    if sha256(source_path(root, row["geotiff_path"])) != row["source_sha256"]:
        raise ValueError("Source changed during processing")
    return {"processed_filename": str(paths[0].relative_to(output)),
            "mask_filename": str(paths[1].relative_to(output)), "scl_filename": str(paths[2].relative_to(output)),
            "width": profile["width"], "height": profile["height"],
            "resolution_x_m": abs(profile["transform"].a), "resolution_y_m": abs(profile["transform"].e),
            "crs": profile["crs"].to_string(), "valid_pixel_count": int(valid.sum()),
            "invalid_pixel_count": int((~valid).sum()), "status": "reused" if cached else "processed",
            "scl_counts": json.dumps(dict(zip(*[x.tolist() for x in np.unique(scl, return_counts=True)])), sort_keys=True)}, (reflectance, analysis, valid)


def rgb_display(bands):
    rgb = np.moveaxis(bands[[2, 1, 0]], 0, -1)
    low, high = DISPLAY_RANGE
    return np.clip((rgb - low) / (high - low), 0, 1)


def comparison_figure(root, output, row, arrays, smoothing=None):
    """Show creator PNG beside fixed-range reflectance and masked analysis RGB."""
    reflectance, analysis, valid = arrays
    png = source_path(root, row["png_path"])
    figure, axes = plt.subplots(1, 5 if smoothing else 4, figsize=(18 if smoothing else 15, 4.5))
    try:
        try:
            axes[0].imshow(plt.imread(png))
        except (OSError, ValueError):
            axes[0].text(.5, .5, "Source PNG unavailable", ha="center")
        axes[0].set_title("Original PNG\nCreator's site-wide stretch")
        axes[1].imshow(rgb_display(reflectance))
        axes[1].set_title("Before: scaled reflectance RGB\nFixed display range 0–0.6")
        axes[2].imshow(~valid, cmap=ListedColormap(["white", "red"]), vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"Invalid mask (red)\n{int((~valid).sum())} invalid pixels")
        display = rgb_display(analysis)
        display[~valid] = [1, 0, 1]
        axes[3].imshow(display)
        axes[3].set_title("After: prepared analysis RGB\nNaN shown in magenta")
        if smoothing:
            smoothed = rgb_display(smooth_for_display(analysis, method=smoothing))
            smoothed[~valid] = [1, 0, 1]
            axes[4].imshow(smoothed)
            axes[4].set_title(f"Optional {smoothing} preview\nNot used in scientific exports")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{row['location_name']} — {row['year']} ({row['acquisition_date']})",
                        y=0.96, fontsize=13)
        figure.subplots_adjust(top=0.76, bottom=0.02, left=0.01, right=0.99, wspace=0.05)
        destination = output / "visualizations" / f"{Path(row['geotiff_path']).stem}_comparison.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    finally:
        plt.close(figure)


def run(root: Path, output: Path, limit=None, smoothing=None):
    root, output = root.resolve(), output.resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Source and output directories must be separate and non-nested")
    records, issues = inspect_sources(root)
    output.mkdir(parents=True, exist_ok=True)
    selected = records if limit is None else records[:limit]
    report, figure_count = [], 0
    for number, row in enumerate(selected, start=1):
        result = {"location_id": row["location_id"], "location": row["location_name"], "year": row["year"],
                  "acquisition_date": row["acquisition_date"], "original_filename": row["geotiff_path"],
                  "source_sha256": row.get("source_sha256", ""), "status": "failed",
                  "message": "; ".join(row["errors"]), "warnings": "; ".join(row["warnings"])}
        if not row["errors"]:
            try:
                details, arrays = export_observation(root, output, row)
                result.update(details)
                # Include temporal endpoints, a typical dune site and both flagged chips.
                representative = (row["location_id"], int(row["year"])) in {
                    ("L1", 2019), ("L1", 2025), ("L1", 2026), ("L10", 2025), ("L21", 2026)}
                if representative or limit == 1:
                    try:
                        comparison_figure(root, output, row, arrays, smoothing)
                        figure_count += 1
                    except (OSError, ValueError) as error:
                        result["warnings"] += f"; Visualization failed: {error}"
                        issues.append({"severity": "warning", "file": row["geotiff_path"], "issue": str(error)})
            except (OSError, ValueError, rasterio.errors.RasterioError) as error:
                result["status"], result["message"] = "failed", str(error)
                issues.append({"severity": "error", "file": row["geotiff_path"], "issue": str(error)})
        report.append(result)
        if number % 25 == 0 or number == len(selected):
            print(f"Checked/exported {number}/{len(selected)} records", flush=True)
    frame = pd.DataFrame(report)
    frame.to_csv(output / "preprocessing_report.csv", index=False)
    pd.DataFrame(issues, columns=["severity", "file", "issue"]).to_csv(output / "validation_issues.csv", index=False)
    statuses = Counter(frame["status"])
    successful = frame[frame["status"].isin(["processed", "reused"])]
    summary = {"metadata_records": len(records), "selected_records": len(selected),
               "expected_records": 200, "processed": statuses["processed"], "reused": statuses["reused"],
               "failed": statuses["failed"], "inventory_errors": sum(i["severity"] == "error" for i in issues),
               "warnings": sum(i["severity"] == "warning" for i in issues), "comparison_figures": figure_count,
               "valid_pixels": int(successful.get("valid_pixel_count", pd.Series(dtype=int)).sum()),
               "invalid_pixels": int(successful.get("invalid_pixel_count", pd.Series(dtype=int)).sum()),
               "filtering_scientific_exports": "disabled", "preview_smoothing": smoothing or "disabled",
               "invalid_scl_classes": list(INVALID_SCL), "partial_run": limit is not None}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text(output_documentation(summary))
    print(json.dumps(summary, indent=2))
    return summary


def output_documentation(summary):
    return f'''# Processed Dataset

Generated by `preprocessing.py` from the read-only source dataset. This run selected
{summary['selected_records']} of {summary['metadata_records']} records; exported {summary['processed']},
reused and verified {summary['reused']}, failed {summary['failed']}. Partial run: {summary['partial_run']}.
Inventory/validation errors: {summary['inventory_errors']}; warnings: {summary['warnings']}.
Valid pixels: {summary['valid_pixels']:,}; invalid pixels: {summary['invalid_pixels']:,}.

## Operations and formats

- `geotiff/<year>/*_processed.tif`: six Float32 surface-reflectance bands in order
  B02, B03, B04, B08, B11, B12. Source UInt16 / 10,000; invalid pixels are NaN
  in all six bands and NaN is declared nodata. No quantitative clipping, stretch,
  normalization, interpolation, warping, shifting or filtering is applied.
- `masks/<year>/*_valid.tif`: UInt8 binary mask (1 valid, 0 invalid), no nodata
  sentinel, so both values are readable. Invalid means SCL 0 (no data), 1
  (saturated/defective), 3 (shadow), 8/9 (cloud), 10 (cirrus), 11 (snow/ice),
  or an invalid Rasterio source mask. SCL 2/4/5/6/7 remain valid; unclassified
  pixels are not automatically clouds. Valid does not imply dune or sand.
- `masks/<year>/*_scl.tif`: unchanged SCL categories as UInt8; no nodata sentinel,
  so class 0 is preserved explicitly. Original source SCL is never filtered.
- `visualizations/*_comparison.png`: source PNG, scaled reflectance RGB,
  invalid mask, prepared RGB. Reflectance panels use the same fixed 0–0.6
  display range across all sites/years; clipping is display-only. Invalid pixels
  are magenta. Creator PNGs use their existing location-wide stretch and are
  reference views, not numerical inputs. Preview smoothing: {summary['preview_smoothing']}.
- `preprocessing_report.csv`: per-record paths, acquisition date, dimensions,
  resolution, CRS, source hash, valid/invalid counts, SCL counts, status and errors.
  `reused` means existing outputs were reopened and verified against source arrays.
- `validation_issues.csv`: missing/corrupt files, metadata/hash mismatches,
  duplicate records, unlisted GeoTIFFs, temporal grid errors and PNG warnings.
- `run_summary.json`: machine-readable results of this invocation. Reports describe
  the current invocation; an explicit `--limit` run is partial and may leave prior
  exports on disk. Run without `--limit` for the complete report.

Dimensions, CRS, affine transforms, bounds, spectral band descriptions and source
provenance are preserved. Every export is reopened and compared with its expected
values, masks and spatial grid. Entire location groups with differing temporal
grids are rejected. No alignment correction is performed. Source SHA-256 values
must match metadata, and are rechecked after processing. Identical pixel grids
do not prove that ground features are stationary or radiometrically equivalent.

## Use by teammates

```python
from pathlib import Path
import numpy as np
import rasterio

base = Path("Processed Dataset")
with rasterio.open(base / "geotiff/2020/L1_Abqaiq_N_2020_processed.tif") as ds:
    reflectance = ds.read()  # (6, 400, 400); ALREADY scaled, do not divide again
    profile = ds.profile
with rasterio.open(base / "masks/2020/L1_Abqaiq_N_2020_valid.tif") as ds:
    valid = ds.read(1).astype(bool)
# Use valid pixels and NaN-aware statistics, e.g. np.nanmean(reflectance, axis=(1, 2)).
```

To access unmasked scientific reflectance without changing originals, use
`preprocessing.load_observation(source_path)`, which returns unmasked reflectance,
SCL, validity, NaN-masked analysis, source profile, descriptions, and tags.
The unmasked array and original files preserve reflectance at SCL-flagged pixels.

## Scientific limitations and provenance

According to `RawDataset/README.md`, the source is Copernicus Sentinel-2 L2A,
retrieved through Sentinel Hub on CDSE: 25 sites × 8 years (2019–2026), 400×400
pixels, EPSG:32639, 10 m grid, six spectral bands and SCL. The creators already
selected dates/sites, used L2A atmospheric correction and Sentinel Hub baseline
harmonization, extracted chips, upsampled native 20 m SWIR bands by nearest
neighbour, co-registered requested grids, and generated PNG visualizations.
This script only validates, converts stored reflectance, masks and exports.

SCL can mistake bright sand for cloud; flagged pixels are retained in source and
unmasked arrays but conservatively excluded from analysis. The source documentation
identifies four flagged pixels each in L1/2025 and L10/2025 as possible examples.
This is a limitation, not a new cloud-detection result. No new cloud classifier
or ground-truth assessment is implemented. Sensor/date variation, residual haze,
native 20 m SWIR detail, annual sampling, site selection bias and absent labels
remain. Geospatial grid agreement does not eliminate these limitations.

Optional `smooth_for_display` supports Gaussian or NaN-aware median smoothing on
a separate copy and retains invalid pixels. It is disabled by default, affects
only an optional preview, and never affects scientific exports. Smoothing can
remove small dune features and alter measurements. No segmentation, temporal
change detection, or evaluation is implemented.

Contains modified Copernicus Sentinel data [2019–2026], processed by Sentinel Hub
via the Copernicus Data Space Ecosystem. See the source README for full attribution.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Source folder (default: RawDataset, or legacy dataset)")
    parser.add_argument("--limit", type=int, help="Process first N metadata rows; still preflight all sources")
    parser.add_argument("--preview-smoothing", choices=["gaussian", "median"], help="Optional display-only smoothing")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    project = Path(__file__).resolve().parent
    source = args.source or (project / "RawDataset" if (project / "RawDataset").exists() else project / "dataset")
    try:
        summary = run(source, project / "Processed Dataset", args.limit, args.preview_smoothing)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        return 1
    return int(bool(summary["failed"] or summary["inventory_errors"]))


if __name__ == "__main__":
    sys.exit(main())
