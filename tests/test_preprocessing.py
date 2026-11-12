"""Scientific invariants and failure cases; all fixtures use temporary folders."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

import preprocessing as prep


class ArrayTests(unittest.TestCase):
    def test_scaling_and_all_scl_classes_without_clipping_or_mutation(self):
        raw = np.full((7, 1, 12), 12000, dtype=np.uint16)
        raw[6] = np.arange(12)
        original = raw.copy()
        reflectance, scl, valid, analysis = prep.prepare_arrays(raw, np.ones((1, 12), bool))
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(scl, np.arange(12)[None])
        np.testing.assert_array_equal(valid[0], [False, False, True, False, True, True, True, True, False, False, False, False])
        np.testing.assert_array_equal(reflectance, np.full((6, 1, 12), np.float32(1.2)))
        self.assertEqual(reflectance.dtype, np.float32)
        self.assertTrue(np.isnan(analysis[:, ~valid]).all())
        np.testing.assert_array_equal(analysis[:, valid], reflectance[:, valid])
        analysis[0, 0, 2] = 0
        self.assertEqual(reflectance[0, 0, 2], np.float32(1.2))

    def test_source_nodata_mask_overrides_clear_scl(self):
        raw = np.full((7, 2, 2), 5000, dtype=np.uint16)
        raw[6] = 5
        mask = np.array([[True, False], [True, True]])
        _, _, valid, analysis = prep.prepare_arrays(raw, mask)
        np.testing.assert_array_equal(valid, mask)
        self.assertTrue(np.isnan(analysis[:, 0, 1]).all())

    def test_smoothing_preserves_invalid_pixels_input_and_band_separation(self):
        image = np.stack([np.full((5, 5), i, np.float32) for i in range(6)])
        image[:, 2, 2] = np.nan
        original = image.copy()
        for method in ("gaussian", "median"):
            smoothed = prep.smooth_for_display(image, method, 3)
            np.testing.assert_array_equal(image, original)
            np.testing.assert_allclose(smoothed, original, equal_nan=True)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "raw"
        self.output = Path(self.temp.name) / "processed"
        self.root.mkdir()

    def fixture(self, year=2019, offset=0):
        name = "L1_Test"
        path = self.root / "geotiff" / str(year) / f"{name}_{year}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        transform = from_origin(360000 + offset, 2880000, 10, 10)
        raw = np.full((7, 400, 400), 3200, dtype=np.uint16)
        raw[6] = 5
        raw[6, 0, 0] = 9
        with rasterio.open(path, "w", driver="GTiff", width=400, height=400,
                           count=7, dtype="uint16", crs="EPSG:32639", transform=transform) as ds:
            ds.write(raw)
            ds.descriptions = (*prep.BANDS, "SCL")
            ds.update_tags(location=name, acquisition_date=f"{year}-03-04")
            bounds = ds.bounds
        return {"location_id": "L1", "location_name": name, "year": year,
                "acquisition_date": f"{year}-03-04", "geotiff_path": str(path.relative_to(self.root)),
                "png_path": f"images/{year}/{name}_{year}.png", "geotiff_sha256": prep.sha256(path),
                "width_px": 400, "height_px": 400, "resolution_m": 10, "crs": "EPSG:32639",
                **dict(zip(("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"), bounds))}

    def inspect(self, rows):
        pd.DataFrame(rows).to_csv(self.root / "metadata.csv", index=False)
        return prep.inspect_sources(self.root)

    def test_export_roundtrip_cache_and_source_preservation(self):
        row = self.fixture()
        records, _ = self.inspect([row])
        self.assertFalse(records[0]["errors"])
        result, _ = prep.export_observation(self.root, self.output, records[0])
        self.assertEqual(result["invalid_pixel_count"], 1)
        self.assertEqual(result["valid_pixel_count"], 159999)
        self.assertEqual(result["status"], "processed")
        self.assertEqual(prep.sha256(self.root / row["geotiff_path"]), row["geotiff_sha256"])
        again, _ = prep.export_observation(self.root, self.output, records[0])
        self.assertEqual(again["status"], "reused")
        output = prep.output_paths(self.output, row)[0]
        with rasterio.open(output, "r+") as ds:
            pixels = ds.read()
            pixels[0, 1, 1] = 999  # Damaged cached output must be regenerated.
            ds.write(pixels)
        repaired, _ = prep.export_observation(self.root, self.output, records[0])
        self.assertEqual(repaired["status"], "processed")

    def test_temporal_grid_mismatch_rejects_entire_location(self):
        records, _ = self.inspect([self.fixture(), self.fixture(2020, offset=10)])
        self.assertTrue(all(any("Temporal grid mismatch" in e for e in r["errors"]) for r in records))

    def test_duplicate_missing_corrupt_and_unlisted_sources_are_reported(self):
        first, missing, corrupt = self.fixture(), self.fixture(2020), self.fixture(2021)
        (self.root / missing["geotiff_path"]).unlink()
        (self.root / corrupt["geotiff_path"]).write_bytes(b"not a TIFF")
        corrupt["geotiff_sha256"] = prep.sha256(self.root / corrupt["geotiff_path"])
        self.fixture(2022)  # Deliberately absent from metadata.
        records, issues = self.inspect([first, first.copy(), missing, corrupt])
        self.assertTrue(all(row["errors"] for row in records))
        self.assertTrue(any("no metadata record" in issue["issue"] for issue in issues))
        self.assertTrue(any("Duplicate" in error for error in records[0]["errors"]))

    def test_hash_mismatch_and_path_escape_are_rejected(self):
        row = self.fixture()
        row["geotiff_sha256"] = "incorrect"
        records, _ = self.inspect([row])
        self.assertIn("SHA-256", records[0]["errors"][0])
        with self.assertRaises(ValueError):
            prep.source_path(self.root, "../outside.tif")

    def test_foreign_output_is_not_overwritten(self):
        row = self.fixture()
        records, _ = self.inspect([row])
        output = prep.output_paths(self.output, row)[0]
        output.parent.mkdir(parents=True)
        original = self.root / row["geotiff_path"]
        output.write_bytes(original.read_bytes())
        checksum = prep.sha256(output)
        with self.assertRaisesRegex(ValueError, "non-pipeline"):
            prep.export_observation(self.root, self.output, records[0])
        self.assertEqual(prep.sha256(output), checksum)


if __name__ == "__main__":
    unittest.main()
