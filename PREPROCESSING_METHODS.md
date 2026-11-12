# Section IV: Data Acquisition and Preprocessing

## Source imagery and preparation performed by the dataset creators

The project dataset comprises 200 Copernicus Sentinel-2 Level-2A observations
covering 25 fixed locations along and around the Abqaiq–Salasil highway in the
Eastern Province of Saudi Arabia, with one observation per location per year
from 2019 through 2026. Each chip covers 4 × 4 km and contains 400 × 400 pixels
on a 10 m grid in WGS 84 / UTM zone 39N (EPSG:32639). Six spectral bands are
provided: B02, B03, B04, B08, B11 and B12, together with the Sen2Cor Scene
Classification Layer (SCL). These source and acquisition details are documented
in `RawDataset/README.md` and `RawDataset/metadata.csv`; this task did not repeat
the imagery acquisition or independently verify the remote Copernicus catalogue.

The dataset creators retrieved imagery through the Sentinel Hub APIs on the
Copernicus Data Space Ecosystem. Their source products already include Level-2A
atmospheric correction and the Sentinel Hub handling of processing-baseline
offsets. The creators selected acquisitions and sites, extracted spatial chips,
requested consistent UTM grids, and upsampled the native 20 m B11/B12 bands by
nearest-neighbour interpolation to the common 10 m grid. The stored UInt16 spectral
values represent surface reflectance multiplied by 10,000. Creator-generated
true-colour PNGs use a location-specific stretch derived jointly across years;
these PNGs are visualization products and were not used as scientific inputs.

## New loading, validation and preprocessing implementation

The new Python pipeline uses Pandas to read the metadata table, Rasterio to load
GeoTIFFs and NumPy for numerical processing. The metadata paths match each image
to its location, acquisition date and year. Validation checks readability,
source SHA-256 against metadata, duplicate observation keys, paths and recorded
hashes, filename conventions, embedded location/date tags, band descriptions,
seven UInt16 bands, 400 × 400 dimensions, 10 m resolution, EPSG:32639, and
metadata bounding boxes and affine transforms. It also inventories GeoTIFFs not
listed in metadata and reports missing or malformed PNG references separately.
For every location, dimensions, CRS, affine transform and bounds are compared
across all available years before export. Mismatches are reported rather than
corrected; no resizing, shifting, registration or warping is performed.

The first six bands are converted to Float32 surface reflectance by division
by 10,000. The unmasked reflectance array preserves all scaled source values
without clipping or contrast enhancement. SCL is copied separately as categorical
UInt8 data. A validity mask excludes SCL classes 0 (no data), 1 (saturated or
defective), 3 (cloud shadow), 8/9 (cloud), 10 (cirrus) and 11 (snow/ice), together
with pixels invalid in any source raster band mask. Classes 2, 4, 5, 6 and 7
remain valid. This is a conservative analysis policy, not a new cloud classifier.
All six bands in the analysis array are assigned NaN at invalid pixels, while
the unmasked array and original GeoTIFF retain their scientific values.

Scientific exports contain six Float32 bands with NaN nodata, the original
spectral descriptions and georeferencing, and source/procedure provenance tags.
Validity masks (1 valid, 0 invalid) and original SCL categories are exported as
separate UInt8 GeoTIFFs. Outputs are organized by year under
`Processed Dataset/geotiff` and `Processed Dataset/masks`. Each file is reopened
and compared against the expected arrays, CRS, transform, dimensions, band
descriptions and type. Content-based caching only reuses outputs after this
verification. Source checksums are checked again after each export.

Comparison figures show the original PNG, scaled reflectance RGB, invalid mask
and prepared analysis RGB. All reflectance panels use the same fixed 0–0.6
display interval, with clipping restricted to the rendered figure. No independent
per-image histogram enhancement is used. Optional Gaussian and median smoothing
functions operate on separate copies, preserve invalid pixels and do not mix
spectral bands. Filtering was disabled for the delivered run and is never applied
to scientific exports. Smoothing could remove small dune features or alter
reflectance measurements.

## Observed results and limits

The complete executed run exported and reopened 200 six-band reflectance
GeoTIFFs, 200 binary validity masks and 200 separate SCL GeoTIFFs. All 25 locations
retain matching spatial grids across their eight annual observations. There
were no source checksum, inventory, metadata or grid errors, and no processing
failures. Across the 32,000,000 spatial pixels, 31,999,992 were valid and 8 were
excluded. Four SCL class-9 pixels occur in each of L1/2025 and L10/2025; the other
198 observations have no pixels excluded by this policy. These are pixel counts,
not segmentation accuracy or independently confirmed cloud detections.

The source documentation identifies the flagged pixels as possibly bright sand
misclassified by SCL. They are conservatively excluded in the delivered analysis
arrays and remain available in the original data and unmasked loading function.
This task does not establish their actual cloud status. Residual radiometric
variation, haze, mixed Sentinel-2 sensors, annual sampling and date differences,
native 20 m SWIR resolution, site selection bias and missing ground-truth labels
remain limitations. Matching spatial grids do not establish radiometric
equivalence or feature stability. No normalization, segmentation, temporal change
detection or evaluation algorithms were implemented in this component.

The executed checks and counts are recorded in
`Processed Dataset/preprocessing_report.csv`, `validation_issues.csv` and
`run_summary.json`. Five sample comparison figures are supplied. Eight automated
tests cover scaling, masking, nonmutation, optional filtering, export/cache
verification, temporal-grid rejection, source errors and protection of files
belonging to other tools. A separate full-dataset check verifies all exported
arrays, masks, SCL categories and temporal grids and checks every original source
file against a pre-implementation SHA-256 manifest.

Source attribution: Contains modified Copernicus Sentinel data [2019–2026],
processed by Sentinel Hub via the Copernicus Data Space Ecosystem. Retain the full
source attribution and citation in `RawDataset/README.md` when using these data.
