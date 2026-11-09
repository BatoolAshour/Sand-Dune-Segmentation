# Abqaiq–Salasil Sentinel-2 Time Series Dataset (2019–2026)

**Course:** ARTI 403 – Image Processing
**Content:** 200 co-registered Sentinel-2 Level-2A image chips: 25 fixed locations × 8 years (2019–2026)
**Intended use:** sand dune segmentation and temporal change detection along the Abqaiq–Salasil highway (Eastern Province, Saudi Arabia)

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

The highway runs along the western edge of a dune field, with farms, settlements and sand sheets on its south-western side.
Sites were placed in two stages.

**Stage 1 (L1–L4).** Four sites were placed by visual inspection of a low-resolution (≈ 60 m) overview of the corridor
(3 March 2023), at points where sand or dunes lie directly beside the road. Each centre was snapped to the nearest point on the KMZ road line.

**Stage 2 (L5–L25).** 21 further sites were selected automatically, using a second ≈ 60 m overview (3 March 2023) of the wider area:
- Candidate chip centres were generated every 250 m along the road, either on the road itself or at a fixed perpendicular offset from it:
  - **near rows:** 4.4–6.4 km NE or SW
  - **far rows:** 8.4–10.4 km NE or SW
- A candidate was accepted only if:
  - it did not overlap any chip already selected (chips may touch);
  - it sat beside the road, not beyond either end (its nearest road point is ≥ 1.5 km from both ends);
  - at least **50 % of its pixels were sand-like** in the overview (red/blue reflectance ratio ≥ 1.85 and mean visible reflectance ≥ 0.30).
- Rows were filled in this order until 25 sites were reached: road, near NE, near SW, far NE, far SW.

All centres are rounded to the 10 m UTM grid, and each chip is a 4 × 4 km box around its centre. The two overview images were used only
for site selection and are **not** part of this dataset.

| Group (`site_row`) | Sites | Distance from road | Typical content |
|---|---|---|---|
| `road` | L1–L6 | on the road (≤ 4 m) | highway with adjacent dunes, sand sheets, farms and junctions |
| `dune_NE` | L7–L12 | 4.4–6.4 km NE | dune field (barchans, linear dunes), some pipelines and tracks |
| `sand_SW` | L13–L17 | 4.2–5.6 km SW | sand sheets and dunes mixed with farms and settlements |
| `dune_NE_far` | L18–L22 | 8.4–10.4 km NE | dense dune field |
| `sand_SW_far` | L23–L25 | 8.8–10.4 km SW | sand sheets with settlements and farms |

Site names encode the group and the position along the road, e.g. `L9_DuneNE_km10_3` is a near-NE dune site opposite road
km 10.3 (measured from the Abqaiq end). Centre coordinates, bounding boxes and distances for every site are in `metadata.csv`.

| ID | Name | Centre (lat, lon) | Road km | Distance from road |
|----|------|-------------------|----|----|
| L1 | L1_Abqaiq_N | 25.988664, 49.636570 | 7.8 | 3 m |
| L2 | L2_Mid_SE | 26.012920, 49.586532 | 13.8 | 0 m |
| L3 | L3_Mid_NW | 26.079991, 49.522646 | 23.8 | 4 m |
| L4 | L4_Salasil_S | 26.114512, 49.473913 | 30.2 | 2 m |
| L5 | L5_Road_km36_0 | 26.150448, 49.432434 | 36.0 | 0 m |
| L6 | L6_Road_km41_2 | 26.113270, 49.433931 | 41.2 | 1 m |
| L7 | L7_DuneNE_km04_7 | 25.981154, 49.703583 | 4.7 | 6.0 km |
| L8 | L8_DuneNE_km08_3 | 26.017622, 49.682898 | 8.3 | 5.6 km |
| L9 | L9_DuneNE_km10_3 | 26.054369, 49.653701 | 10.3 | 6.4 km |
| L10 | L10_DuneNE_km11_9 | 26.049346, 49.609483 | 11.9 | 4.4 km |
| L11 | L11_DuneNE_km27_0 | 26.136793, 49.524232 | 27.0 | 4.4 km |
| L12 | L12_DuneNE_km32_7 | 26.163525, 49.481779 | 32.7 | 4.4 km |
| L13 | L13_SandSW_km04_9 | 25.950628, 49.605352 | 4.9 | 4.4 km |
| L14 | L14_SandSW_km14_1 | 25.970451, 49.555379 | 14.1 | 5.6 km |
| L15 | L15_SandSW_km18_2 | 26.008238, 49.524544 | 18.2 | 4.4 km |
| L16 | L16_SandSW_km36_4 | 26.142755, 49.389728 | 36.4 | 4.2 km |
| L17 | L17_SandSW_km42_7 | 26.095120, 49.392481 | 42.7 | 4.4 km |
| L18 | L18_DuneNEfar_km07_5 | 26.018091, 49.735352 | 7.5 | 10.4 km |
| L19 | L19_DuneNEfar_km09_0 | 26.054546, 49.702576 | 9.0 | 9.6 km |
| L20 | L20_DuneNEfar_km10_7 | 26.092694, 49.658562 | 10.7 | 10.4 km |
| L21 | L21_DuneNEfar_km17_3 | 26.090303, 49.615697 | 17.3 | 8.4 km |
| L22 | L22_DuneNEfar_km24_5 | 26.127741, 49.612656 | 24.5 | 10.4 km |
| L23 | L23_SandSWfar_km01_8 | 25.906117, 49.559058 | 1.8 | 10.4 km |
| L24 | L24_SandSWfar_km15_3 | 25.961733, 49.514638 | 15.3 | 8.8 km |
| L25 | L25_SandSWfar_km18_8 | 25.978717, 49.472374 | 18.8 | 10.4 km |

---

## 3. Image Retrieval and Acquisitions

**Scene selection.** For each year, the Catalog API listed all Sentinel-2 L2A scenes over the area from January to March.
One acquisition per year was chosen. It had to be:
- from the **same relative orbit (R006, ≈ 07:32 UTC overpass)**, for consistent viewing geometry;
- as close as possible to **3–7 March**, for the same season;
- **cloud-free** over all 25 sites. Before any full download, this was confirmed with a low-resolution (40 m) cloud-mask-only (SCL) request per date.

**Retrieval.** One Process API request was made per chip, restricted to the selected acquisition day. Requests used:
- the chip's exact UTM bounding box (EPSG:32639);
- 10 m output pixels (400 × 400);
- nearest-neighbour upsampling for the 20 m SWIR bands.

All 25 sites of a given year come from the same acquisition.

| Year | Date | Day of year | Satellite | Orbit | Processing baseline | Granule cloud (T39RUJ / T39RUK) | Reason for date |
|------|------|----|-------------|------|-------|-------------|----|
| 2019 | 2019-03-04 | 63 | Sentinel-2A | R006 | N0500 | 0.1 / 0.1 % | target window |
| 2020 | 2020-03-03 | 63 | Sentinel-2B | R006 | N0500 | 0.0 / 0.1 % | target window |
| 2021 | 2021-03-08 | 67 | Sentinel-2B | R006 | N0500 | 0.1 / 0.3 % | 3 Mar was 45–89 % cloud |
| 2022 | 2022-03-03 | 62 | Sentinel-2B | R006 | N0510 | 1.6 / 3.1 % | target window; SCL check confirmed all 25 sites clear |
| 2023 | 2023-03-03 | 62 | Sentinel-2A | R006 | N0510 | 0.3 / 0.1 % | target window |
| 2024 | 2024-02-26 | 57 | Sentinel-2A | R006 | N0510 | 0.4 / 0.9 % | 2, 7 and 12 Mar were 33–100 % cloud; nearest clear R006 pass |
| 2025 | 2025-03-02 | 61 | Sentinel-2C | R006 | N0511 | 0.0–0.1 / 0.1 % | target window |
| 2026 | 2026-03-07 | 66 | Sentinel-2C | R006 | N0512 | 1.3 / 0.0 % | 2 Mar was 66–97 % cloud |

All acquisitions fall within a 10-day seasonal window (day of year 57–67). The catalog lists two versions of the 2025 T39RUJ
granule, and both appear in `metadata.csv`.

---

## 4. Dataset Structure

```
dataset/
├── images/                 true-colour PNGs (visualisation)
│   ├── 2019/ 2020/ 2021/ 2022/ 2023/ 2024/ 2025/ 2026/
│   │     L1_Abqaiq_N_<year>.png … L25_SandSWfar_km18_8_<year>.png   (25 per year)
├── geotiff/                georeferenced multiband GeoTIFFs (analysis)
│   ├── 2019/ … 2026/
│   │     L1_Abqaiq_N_<year>.tif … L25_SandSWfar_km18_8_<year>.tif   (25 per year)
├── metadata.csv            one row per image (200 rows)
└── README.md
```

That gives **200 images = 25 locations × 8 years**. Each image is stored once as a PNG and once as a GeoTIFF (400 image files),
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
- location ID, name and site group (`site_row`);
- centre coordinates (lat/lon and UTM), bounding box, position along the road and distance to the road;
- acquisition date, time and day of year;
- satellite, product level, relative orbit, processing baseline and source granule IDs;
- CRS, resolution and image size;
- band list and scaling;
- PNG stretch limits;
- granule-level and chip-level (SCL) cloud and shadow percentages, cloud pixel count, and no-data percentage;
- file paths and a SHA-256 checksum for each GeoTIFF.

---

## 5. Preprocessing

1. **Atmospheric correction:** none applied by us. Level-2A is ESA's bottom-of-atmosphere reflectance product (Sen2Cor).
2. **Radiometric scaling:** reflectance delivered by Sentinel Hub (which harmonises the processing baseline ≥ 04.00 offset),
   stored as reflectance × 10 000. 2019–2026 values are therefore on the same scale.
3. **Spatial subsetting and resampling:** done by the Process API, onto a fixed 10 m UTM 39N grid per site. Only B11/B12 are
   resampled (20 m → 10 m, nearest neighbour).
4. **Co-registration:** every year of a given site is requested on the identical grid, so all eight GeoTIFFs of a site share the
   same CRS, bounding box, affine transform and dimensions. No further warping was applied.
5. **PNG rendering:** true colour (B04/B03/B02) with a linear stretch computed once per location over all eight years
   (1st–99.5th percentile; limits in `png_stretch_min` / `png_stretch_max`). Brightness is therefore comparable across years.
6. **Cloud assessment:** SCL classes 8, 9, 10 (cloud) and 3 (shadow), counted within each chip.
7. **No synthetic data:** no images were generated, interpolated, gap-filled or composited, and no labels were created.

### Quality checks (all 2012 checks passed)

- All 200 GeoTIFFs and 200 PNGs exist, can be read, and match `metadata.csv` exactly.
- Embedded date and location tags match the filenames, `metadata.csv` and the Copernicus catalog granule IDs.
- The 25 site footprints do not overlap. Road sites are centred on the KMZ road line (≤ 4 m), and offset sites lie at their recorded distance from it.
- All eight years of each site have identical grids. Each image correlates best with its own site's 2019 image, not with any other site.
- **Registration:** misregistration would affect every site on the same date, because all sites in a year come from the same scene.
  In every year, all road sites and at least 20 of the 25 sites show a 0-pixel phase-correlation shift relative to 2019.
- Every PNG reproduces its GeoTIFF's RGB bands pixel for pixel.
- **Cloud:** 198 chips have 0 cloud, shadow or no-data pixels. `L1_Abqaiq_N_2025` and `L10_DuneNE_km11_9_2025` each have
  4 pixels (0.0025 %) flagged as high-probability cloud, most likely bright sand misclassified by SCL.

---

## 6. Limitations

- **No ground-truth labels.** The dataset contains no sand, dune or road masks. Supervised segmentation requires manual
  annotation (or unsupervised / spectral-index methods). The KMZ corridor polygon is a study-area boundary, not a label.
- **10 m spatial resolution.** Only larger features are resolved: dune fields, sand sheets, encroachment fronts.
  Thin sand on individual road lanes, small drifts and objects under ~10–20 m are not visible. The SWIR bands are natively 20 m.
- **One acquisition per year.** Each year is a single late-winter snapshot. Short-term wind events can move sand between acquisitions.
- **Dunes move between years.** In five dune-field chips (L9, L10, L20, L21, L22) the dominant texture is displaced by 1–6 pixels
  relative to 2019, almost entirely north–south. In L21 and L22 the displacement grows steadily to 5–6 px (50–60 m) southward by 2026.
  Road and farm sites show no shift on the same scenes, so this is surface change (dune migration), not misregistration.
  This is a useful signal for change detection, but dune pixels are not static targets across years.
- **Multiple sensors and dates.** Sentinel-2A, 2B and 2C are mixed, and dates vary by up to 10 days of year (2024-02-26 is the
  earliest, with a slightly lower sun). Residual radiometric differences remain. For example, the 2026 chips of L1–L4 are uniformly darker
  than 2020 (≈ −0.03 mean visible reflectance), which is most likely radiometric rather than surface change.
  Relative normalisation (e.g. histogram matching or pseudo-invariant features) is recommended before change detection.
- **Site selection bias.** Stage 2 sites were chosen with a simple spectral sand rule on one 2023 overview, so the dataset over-represents
  sandy terrain. L1–L4 were chosen visually. Sites are representative of sand-dominated land near the highway, not of the region as a whole.
- **Atmospheric and mask uncertainty.** Residual dust/haze can remain after L2A correction. Sen2Cor's SCL can misclassify very bright
  sand as cloud. Cloud figures rely on SCL plus visual inspection of a subset of chips.
- **PNGs are for display only.** They are contrast-stretched 8-bit images. Use the GeoTIFFs for any quantitative work.

---

## 7. Data Attribution and Citation

**Data access dates:** Catalog and Process API requests were made on **13 October 2026** (L1–L4 for 2020–2024 and 2026; 24 images)
and on **9 November 2026** (all other 176 images, the 2019 and 2025 catalog searches, and the stage-2 overview and cloud checks).

**Required attribution** (Copernicus Sentinel data are free and open under the Copernicus legal notice; modified data must be
attributed as follows):

> Contains modified Copernicus Sentinel data [2019–2026], processed by Sentinel Hub
> via the Copernicus Data Space Ecosystem.

**Suggested citation for this dataset:**

> ARTI 403 Project Team (2026). *Abqaiq–Salasil Sentinel-2 Time Series Dataset (2019–2026)* [Data set].
> https://github.com/BatoolAshour/Sand-Dune-Segmentation. Derived from Copernicus Sentinel-2 Level-2A imagery,
> Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/), accessed 13 October and 9 November 2026.

When using this dataset, cite both the Copernicus source (attribution above) and this dataset. Credit for the underlying imagery
belongs to the Copernicus programme / ESA. This dataset only selects, organises and visualises it.
