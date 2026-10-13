# Abqaiq–Salasil Sentinel-2 Time Series Dataset (2020–2026)

**Course:** ARTI 403 – Image Processing
**Content:** 24 co-registered Sentinel-2 Level-2A image chips: 4 fixed locations × 6 years (2020, 2021, 2022, 2023, 2024, 2026)
**Intended use:** sand segmentation and temporal change detection along the Abqaiq–Salasil highway (Eastern Province, Saudi Arabia)

---

## 1. Data Source and Provenance

All imagery comes from the **Copernicus Sentinel-2 mission** (European Union / ESA). It was retrieved directly from the
**Copernicus Data Space Ecosystem (CDSE)** through the **Sentinel Hub APIs**. The dataset was **not** obtained from
Kaggle, Hugging Face or any other third-party repository.

| Resource | Link |
|---|---|
| Copernicus Data Space Ecosystem | https://dataspace.copernicus.eu/ |
| Copernicus Browser (visual inspection of the same scenes) | https://browser.dataspace.copernicus.eu/ |
| Sentinel Hub API documentation (CDSE) | https://documentation.dataspace.copernicus.eu/APIs/SentinelHub.html |

APIs used:
- **Catalog API**: searched the scene metadata (dates, granule IDs, cloud cover) to select acquisitions. No imagery is downloaded at this step.
- **Process API**: retrieved the pixel data for each 4 × 4 km chip as a GeoTIFF.

Access used the project's CDSE OAuth client credentials. Credentials are not part of this dataset.

### Original data vs. this dataset

| | Original Copernicus data | This dataset (custom-organised) |
|---|---|---|
| Producer | ESA / Copernicus programme | ARTI 403 project team |
| Product | Sentinel-2 MSI Level-2A granules (100 × 100 km MGRS tiles T39RUJ and T39RUK) | 4 × 4 km chips extracted from those granules by the Sentinel Hub Process API |
| Content | All 13 spectral bands plus auxiliary layers | 6 spectral bands + scene classification layer (SCL) |
| Our additions | – | Site selection, file organisation and naming, true-colour PNG renderings, `metadata.csv`, this README |

The **GeoTIFFs** contain Copernicus reflectance values exactly as delivered by the Process API. We did not alter the pixel values.
The **PNGs** are our own visualisation products, derived from those GeoTIFFs. The original granule IDs for every image are listed
in `metadata.csv` (`source_granules`).

---

## 2. Study Area and Site Selection

The study area is defined by the project file `Abqaiq_Salasil_Route4_2km_Study_Area.kmz`. It contains:
- **"Abqaiq–Salasil route-4"**: the highway centre line (283 vertices, 46.2 km), and
- **"2 km study area each side"**: a corridor polygon extending 2 km on each side of the road.

The four sites were selected as follows:
1. The road line and corridor were read from the KMZ.
2. A single low-resolution (≈ 60 m) true-colour overview of the corridor (3 March 2023) was inspected. The road runs along
   the western edge of a dune field, with farms and settlements on its western side.
3. Four points were placed along the road where sand or dunes lie directly beside it, spread along the route.
4. Each point was snapped to the nearest point on the KMZ road line and rounded to the 10 m UTM grid.
5. A 4 × 4 km chip (≈ 2 km either side of the road, matching the KMZ corridor width) was built around each centre.

The overview image was used only for site selection and is **not** part of this dataset.

| ID | Name | Centre (lat, lon) | Centre UTM 39N (E, N) | Road distance from Abqaiq | Setting |
|----|------|-------------------|-----------------------|----|---------|
| L1 | L1_Abqaiq_N  | 25.988664 N, 49.636570 E | 363530, 2875140 | 7.8 km  | barchan dunes on both sides of the road; settlement to the NE |
| L2 | L2_Mid_SE    | 26.012920 N, 49.586532 E | 358550, 2877880 | 13.8 km | road bend at the edge of the dune field |
| L3 | L3_Mid_NW    | 26.079991 N, 49.522646 E | 352240, 2885380 | 23.8 km | dune/sand sheet to the NE; farms to the SW |
| L4 | L4_Salasil_S | 26.114512 N, 49.473913 E | 347410, 2889260 | 30.2 km | sand sheet and dunes close to the road |

---

## 3. Image Retrieval and Acquisitions

**Scene selection.** For each year, the Catalog API listed all Sentinel-2 L2A scenes over the corridor from January to March.
One acquisition per year was chosen. It had to be:
- from the **same relative orbit (R006, ≈ 07:32 UTC overpass)**, for consistent viewing geometry;
- as close as possible to **3–7 March**, for the same season;
- **cloud-free** over the four sites.

**Retrieval.** One Process API request was made per chip, restricted to the selected acquisition day. Requests used:
- the chip's exact UTM bounding box (EPSG:32639);
- 10 m output pixels (400 × 400);
- nearest-neighbour upsampling for the 20 m SWIR bands.

| Year | Date | Day of year | Time (UTC) | Satellite | Orbit | Processing baseline | Granule cloud (T39RUJ / T39RUK) | Reason for date |
|------|------|----|----------|-------------|------|-------|-------------|----|
| 2020 | 2020-03-03 | 63 | 07:32:31 | Sentinel-2B | R006 | N0500 | 0.0 / 0.1 % | target window |
| 2021 | 2021-03-08 | 67 | 07:32:32 | Sentinel-2B | R006 | N0500 | 0.1 / 0.3 % | 3 Mar was 45–89 % cloud |
| 2022 | 2022-03-03 | 62 | 07:32:30 | Sentinel-2B | R006 | N0510 | 1.6 / 3.1 % | target window; a low-resolution SCL check confirmed all 4 sites cloud-free |
| 2023 | 2023-03-03 | 62 | 07:32:30 | Sentinel-2A | R006 | N0510 | 0.3 / 0.1 % | target window |
| 2024 | 2024-02-26 | 57 | 07:32:32 | Sentinel-2A | R006 | N0510 | 0.4 / 0.9 % | 2, 7 and 12 Mar were 33–100 % cloud; nearest clear R006 pass |
| 2026 | 2026-03-07 | 66 | 07:32:34 | Sentinel-2C | R006 | N0512 | 1.3 / 0.0 % | 2 Mar was 66–97 % cloud |

All acquisitions fall within a 10-day seasonal window (day of year 57–67). **2025 is not included.**

---

## 4. Dataset Structure

```
dataset/
├── images/                 true-colour PNGs (visualisation)
│   ├── 2020/ 2021/ 2022/ 2023/ 2024/ 2026/
│   │     L1_Abqaiq_N_<year>.png  L2_Mid_SE_<year>.png  L3_Mid_NW_<year>.png  L4_Salasil_S_<year>.png
├── geotiff/                georeferenced multiband GeoTIFFs (analysis)
│   ├── 2020/ 2021/ 2022/ 2023/ 2024/ 2026/
│   │     L1_Abqaiq_N_<year>.tif  …  L4_Salasil_S_<year>.tif
├── metadata.csv            one row per image (24 rows)
└── README.md
```

That gives **24 images = 4 locations × 6 years**. Each image is stored once as a PNG and once as a GeoTIFF (48 image files),
and every image is a separate file. Filenames follow `<locationID>_<name>_<year>`.

### PNG vs. GeoTIFF

| | `geotiff/*.tif` | `images/*.png` |
|---|---|---|
| Purpose | **Image processing and analysis** (segmentation, indices, change detection) | **Visualisation**, reports, quick inspection, annotation backdrop |
| Georeferenced | Yes (EPSG:32639) | No |
| Bands | 7 (see below) | 3 (RGB = B04, B03, B02) |
| Values | Surface reflectance × 10 000 (UInt16) | 8-bit, contrast-stretched for display |
| Size | 400 × 400 px | 400 × 400 px (pixel (r, c) is the same location as in the GeoTIFF) |

### GeoTIFF specification

| Property | Value |
|---|---|
| CRS | EPSG:32639 (WGS 84 / UTM zone 39N) |
| Spatial resolution | 10 m × 10 m |
| Extent | 400 × 400 px = 4 km × 4 km; bounding boxes are in `metadata.csv` |
| Data type | UInt16, DEFLATE compression |
| Band 1 | B02 – Blue (490 nm), native 10 m |
| Band 2 | B03 – Green (560 nm), native 10 m |
| Band 3 | B04 – Red (665 nm), native 10 m |
| Band 4 | B08 – Near-infrared (842 nm), native 10 m |
| Band 5 | B11 – SWIR-1 (1610 nm), native 20 m, resampled to 10 m (nearest neighbour) |
| Band 6 | B12 – SWIR-2 (2190 nm), native 20 m, resampled to 10 m (nearest neighbour) |
| Band 7 | SCL – Sen2Cor Scene Classification Layer (class codes; e.g. 3 cloud shadow, 4 vegetation, 5 bare soil, 8/9 cloud, 10 cirrus) |
| Scaling | Bands 1–6: reflectance × 10 000 (divide by 10 000 for reflectance) |

Band names and provenance tags (`acquisition_date`, `location`, `source`) are embedded in each GeoTIFF.

```python
import rasterio
with rasterio.open('geotiff/2020/L1_Abqaiq_N_2020.tif') as ds:
    refl = ds.read([1, 2, 3, 4, 5, 6]).astype('float32') / 10000   # B02, B03, B04, B08, B11, B12
    scl  = ds.read(7)
```

### metadata.csv

There is one row per image. The columns include:
- location ID and name, centre coordinates (lat/lon and UTM), bounding box;
- acquisition date, time and day of year;
- satellite, product level, relative orbit, processing baseline and source granule IDs;
- CRS, resolution and image size;
- band list and scaling;
- PNG stretch limits;
- granule-level and chip-level (SCL) cloud and shadow percentages, and no-data percentage;
- file paths and a SHA-256 checksum for each GeoTIFF.

---

## 5. Preprocessing

1. **Atmospheric correction:** none applied by us. Level-2A is ESA's bottom-of-atmosphere reflectance product (Sen2Cor).
2. **Radiometric scaling:** reflectance delivered by Sentinel Hub (which harmonises the processing baseline ≥ 04.00 offset),
   stored as reflectance × 10 000. 2020–2026 values are therefore on the same scale.
3. **Spatial subsetting and resampling:** done by the Process API, onto a fixed 10 m UTM 39N grid per site. Only B11/B12 are
   resampled (20 m → 10 m, nearest neighbour).
4. **Co-registration:** every year of a given site is requested on the identical grid, so all six GeoTIFFs of a site share the
   same CRS, bounding box, affine transform and dimensions. No further warping was applied.
5. **PNG rendering:** true colour (B04/B03/B02) with a linear stretch computed once per location over all six years
   (1st–99.5th percentile; limits in `png_stretch_min` / `png_stretch_max`). Brightness is therefore comparable across years.
6. **Cloud assessment:** SCL classes 8, 9, 10 (cloud) and 3 (shadow), counted within each chip.
7. **No synthetic data:** no images were generated, interpolated, gap-filled or composited, and no labels were created.

### Quality checks (all passed)

- All 24 GeoTIFFs and 24 PNGs exist and can be read.
- Embedded date and location tags match the filenames, `metadata.csv` and the Copernicus catalog.
- Every chip centre lies on the KMZ road line (≤ 4 m).
- All six years of each site have identical grids. Image phase correlation shows a 0-pixel shift between years.
- Every PNG reproduces its GeoTIFF's RGB bands pixel for pixel.
- All 24 chips: **0.0 % cloud, 0.0 % cloud shadow, 0.0 % no-data** (SCL).

---

## 6. Limitations

- **No ground-truth labels.** The dataset contains no sand, dune or road masks. Supervised segmentation requires manual
  annotation (or unsupervised / spectral-index methods). The KMZ corridor polygon is a study-area boundary, not a label.
- **10 m spatial resolution.** Only larger features are resolved: dune fields, sand sheets, encroachment fronts.
  Thin sand on individual road lanes, small drifts and objects under ~10–20 m are not visible. The SWIR bands are natively 20 m.
- **One acquisition per year, and no 2025.** Each year is a single snapshot, and there is a two-year gap between 2024 and 2026.
  Short-term wind events can move sand between acquisitions.
- **Multiple sensors and dates.** Sentinel-2A, 2B and 2C are mixed, and dates vary by up to 10 days of year (2024-02-26 is the
  earliest, with a slightly lower sun). Residual radiometric differences remain. For example, the 2026 chips are uniformly darker
  than 2020 (≈ −0.03 mean visible reflectance) at all sites, which is most likely radiometric rather than surface change.
  Relative normalisation (e.g. histogram matching or pseudo-invariant features) is recommended before change detection.
- **Atmospheric and mask uncertainty.** Residual dust/haze can remain after L2A correction. Sen2Cor's SCL can misclassify very bright
  sand as cloud. Cloud figures rely on SCL plus visual inspection of a subset of chips.
- **Small sample.** Four 16 km² sites are a pilot dataset and do not represent the entire 46 km route.
- **PNGs are for display only.** They are contrast-stretched 8-bit images. Use the GeoTIFFs for any quantitative work.

---

## 7. Data Attribution and Citation

**Data access dates:** all Catalog and Process API requests were made on **13 October 2026**.

**Required attribution** (Copernicus Sentinel data are free and open under the Copernicus legal notice; modified data must be
attributed as follows):

> Contains modified Copernicus Sentinel data [2020, 2021, 2022, 2023, 2024, 2026], processed by Sentinel Hub
> via the Copernicus Data Space Ecosystem.

**Suggested citation for this dataset:**

> ARTI 403 Project Team (2026). *Abqaiq–Salasil Sentinel-2 Time Series Dataset (2020–2026)* [Data set]. Derived from
> Copernicus Sentinel-2 Level-2A imagery, Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/),
> accessed 13 October 2026.

When using this dataset, cite both the Copernicus source (attribution above) and this dataset. Credit for the underlying imagery
belongs to the Copernicus programme / ESA. This dataset only selects, organises and visualises it.
