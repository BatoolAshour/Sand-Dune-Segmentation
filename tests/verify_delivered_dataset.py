"""Independent, read-only comparison of every delivered raster with its source.

Run from VS Code: ``python tests/verify_delivered_dataset.py``.
Optional ``--source-manifest`` checks all source files against an earlier JSON
mapping of repository-relative paths to SHA-256, including PNGs and documentation.
Only the verification result is written, inside Processed Dataset.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(project, manifest=None):
    source = project / "RawDataset"
    if not source.exists():
        source = project / "dataset"
    output = project / "Processed Dataset"
    metadata = pd.read_csv(source / "metadata.csv")
    report = pd.read_csv(output / "preprocessing_report.csv")
    assert len(metadata) == len(report) == 200
    assert report["status"].isin(["processed", "reused"]).all()
    assert not report.duplicated(["location_id", "year"]).any()
    assert len(list((output / "geotiff").rglob("*.tif"))) == 200
    assert len(list((output / "masks").rglob("*_valid.tif"))) == 200
    assert len(list((output / "masks").rglob("*_scl.tif"))) == 200
    grid_by_location = defaultdict(set)
    flagged, total_invalid = [], 0
    for row in metadata.to_dict("records"):
        path = source / row["geotiff_path"]
        assert digest(path) == row["geotiff_sha256"], f"Source hash changed: {path}"
        stem, year = path.stem, str(row["year"])
        with rasterio.open(path) as original:
            expected = original.read([1, 2, 3, 4, 5, 6]).astype(np.float32) / np.float32(10000)
            scl = original.read(7)
            valid = ~np.isin(scl, [0, 1, 3, 8, 9, 10, 11]) & np.all(original.read_masks() > 0, axis=0)
            expected[:, ~valid] = np.nan
            expected_grid = (original.width, original.height, original.crs, original.transform, original.bounds)
            for relative, array in ((f"geotiff/{year}/{stem}_processed.tif", expected),
                                    (f"masks/{year}/{stem}_valid.tif", valid.astype(np.uint8)[None]),
                                    (f"masks/{year}/{stem}_scl.tif", scl.astype(np.uint8)[None])):
                with rasterio.open(output / relative) as exported:
                    actual_grid = (exported.width, exported.height, exported.crs, exported.transform, exported.bounds)
                    assert actual_grid == expected_grid, relative
                    assert exported.count == array.shape[0], relative
                    assert exported.dtypes == (str(array.dtype),) * array.shape[0], relative
                    np.testing.assert_array_equal(exported.read(), array, err_msg=relative)
                    if "_processed" in relative:
                        assert np.isnan(exported.nodata)
                        assert exported.descriptions == original.descriptions[:6]
                        assert exported.tags()["scale"] == "already scaled; multiplier 1"
            grid_by_location[row["location_id"]].add(expected_grid)
        invalid = int((~valid).sum())
        total_invalid += invalid
        entry = report[(report.location_id == row["location_id"]) & (report.year == row["year"])].iloc[0]
        assert entry.invalid_pixel_count == invalid
        assert entry.valid_pixel_count == int(valid.sum())
        if invalid:
            flagged.append({"location_id": row["location_id"], "year": row["year"], "invalid_pixels": invalid,
                            "invalid_scl_values": np.unique(scl[~valid]).tolist()})
    assert len(grid_by_location) == 25
    assert all(len(grids) == 1 for grids in grid_by_location.values())
    manifest_count = 0
    if manifest:
        before = json.loads(manifest.read_text())
        actual_files = {str(path.relative_to(project)) for path in source.rglob("*") if path.is_file()}
        assert actual_files == set(before), "Source file inventory changed"
        for relative, expected_hash in before.items():
            assert digest(project / relative) == expected_hash, f"Original file changed: {relative}"
        manifest_count = len(before)
    summary = {"verified_reflectance_geotiffs": 200, "verified_validity_masks": 200,
               "verified_scl_geotiffs": 200, "temporally_aligned_locations": len(grid_by_location),
               "original_files_verified_against_before_manifest": manifest_count,
               "valid_pixels": 32000000 - total_invalid, "invalid_pixels": total_invalid,
               "flagged_observations": flagged, "result": "passed"}
    (output / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args()
    verify(Path(__file__).resolve().parents[1], args.source_manifest)
