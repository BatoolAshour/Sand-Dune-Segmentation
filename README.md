# Sand Dune Segmentation and Temporal Change Detection Along the Abqaiq–Salasil Highway Using Satellite Image Processing

ARTI 403 – Image Processing course project, Department of Artificial Intelligence,
Imam Abdulrahman Bin Faisal University, Dammam, Saudi Arabia.

The project maps sand cover at 25 fixed sites along the Abqaiq–Salasil Highway
(Eastern Province, Saudi Arabia) and tracks how it changed between 2019 and 2026,
using 200 free Sentinel-2 Level-2A images. Sand is first segmented with the Bare
Soil Index (BSI) and Otsu thresholding, refined with texture analysis, a SWIR
salt-flat rule and morphological operations, and then compared across years to
map sand gain, loss and persistence.

## Team

| Member | Name | Role | Main files |
|---|---|---|---|
| 1 | Farah Mustafa AlMujaljel | Data loading and preprocessing | [preprocessing.py](preprocessing.py), [tests](tests/), [Processed Dataset](<Processed Dataset/>) |
| 2 | Aryam Hasan Baragash | Sand segmentation (BSI + Otsu) | [Segmentation (1).ipynb](<Segmentation (1).ipynb>), [Segmentation/Results_BSI_Otsu](Segmentation/Results_BSI_Otsu/) |
| 3 | Batool Saeed Ashour | Dataset collection, texture analysis and morphological refinement | [RawDataset](RawDataset/), [texture_morphology.py](texture_morphology.py), [Texture Morphology Results](<Texture Morphology Results/>) |
| 4 | Alaa Wael Bughararah | Temporal change detection and area analysis | [Temporal_Change_Detection.ipynb](Temporal_Change_Detection.ipynb), [Temporal Change Results](<Temporal Change Results/>) |
| 5 | Raneem Omar Bahobail | Evaluation and visualization | [evaluation.py](evaluation.py), [Evaluation Results](<Evaluation Results/>) |

## Key results

| Measure | Value |
|---|---|
| Images analysed | 200 (25 sites × 8 years, 2019–2026) |
| Mean sand cover, BSI + Otsu → refined | 70.66% → 68.36% |
| Sand patches per image, before → after refinement | 1,008 → 11.4 (−98.9%) |
| Full-period sand gain / loss, all 25 sites | 14.88 km² / 51.94 km² |
| Full-period net change | −37.07 km² (20 sites decreased, 5 increased) |
| Largest net increase / decrease | L12: +0.98 km² / L3: −3.69 km² |
| Agreement with BSI + Otsu pseudo-labels (macro mean) | IoU 0.855, Dice 0.920, Precision 0.939, Recall 0.906 |

No hand-labelled ground-truth masks were available, so the IoU, Dice, Precision
and Recall values measure agreement with the BSI + Otsu baseline, not true
segmentation accuracy. At 10 m resolution the temporal results describe changes
in classified sand cover, not the movement of individual dunes.

## Dataset

[RawDataset](RawDataset/) holds 200 co-registered Sentinel-2 Level-2A image chips:
25 fixed 4 × 4 km sites × 8 years (2019–2026), one late-February/early-March
acquisition per year from the same relative orbit (R006). Each chip is
400 × 400 pixels at 10 m, in WGS 84 / UTM 39N (EPSG:32639), with bands
B02, B03, B04, B08, B11, B12 and the Scene Classification Layer (SCL). Six sites
(L1–L6) are centred on the highway; L7–L25 cover nearby and more distant dune
fields and sand sheets to the north-east and south-west.

The imagery was retrieved from the Copernicus Data Space Ecosystem through the
Sentinel Hub APIs. See the [dataset README](RawDataset/README.md) for provenance,
site list, metadata columns and file formats.

## Pipeline

| Stage | Code | Input | Output |
|---|---|---|---|
| 1. Preprocessing | `preprocessing.py` | `RawDataset/` | `Processed Dataset/` (Float32 reflectance, valid-pixel masks, report) |
| 2. Segmentation | `Segmentation (1).ipynb` | `Processed Dataset/` | `Segmentation/Results_BSI_Otsu/` (BSI + Otsu sand masks) |
| 3. Texture and morphology | `texture_morphology.py` | `Processed Dataset/`, BSI + Otsu masks | `Texture Morphology Results/` (refined masks, previews, `summary.csv`) |
| 4. Temporal change | `Temporal_Change_Detection.ipynb` | refined masks | `Temporal Change Results/` (change maps, area tables, figures) |
| 5. Evaluation | `evaluation.py` | outputs of stages 2–4 | `Evaluation Results/` (agreement metrics, tables, figures) |

All masks use the same values: `1` = sand, `0` = not sand, `255` = invalid.

- **Preprocessing** validates every GeoTIFF against `RawDataset/metadata.csv`,
  scales the six bands to Float32 reflectance and masks invalid pixels using SCL.
- **Segmentation** computes the Bare Soil Index from B02, B04, B08 and B11 and
  applies Otsu's threshold to each image to get a potential-sand mask.
- **Texture and morphology** computes GLCM contrast and local entropy in a
  15 × 15 pixel window, removes sand pixels with a texture score above 2 (roads,
  buildings, farms) and pixels with B11/B12 > 1.15 (salt and gypsum flats), then
  cleans the mask with opening (r = 1), closing (r = 2), hole filling (≤ 50 px)
  and removal of patches smaller than 50 px. All settings are in the `Config`
  class at the top of the script.
- **Temporal change** compares masks over pixels valid at both dates and classifies
  persistent sand, persistent non-sand, sand gain and sand loss for 175
  consecutive-year and 25 full-period (2019–2026) comparisons. Each pixel is
  0.0001 km².
- **Evaluation** compares the refined masks with the BSI + Otsu pseudo-labels
  (IoU, Dice, Precision, Recall) and produces the summary tables and figures.

## How to run

Use Python 3.12 or newer. From the repository folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `python -m venv .venv` and `.venv\Scripts\Activate.ps1`. In
VS Code, run **Python: Select Interpreter** and select `.venv`.

Run the stages in order:

```sh
python preprocessing.py                # 1. preprocessing
                                       # 2. run "Segmentation (1).ipynb"
python texture_morphology.py           # 3. texture and morphology
                                       # 4. run Temporal_Change_Detection.ipynb
python evaluation.py                   # 5. evaluation
```

Notes:
- Both notebooks were written for Google Colab and clone this repository at the
  start. The segmentation notebook also needs `scikit-image`
  (`pip install scikit-image`).
- The temporal results are stored as
  `Temporal Change Results/Temporal_Change_Results.zip`. Unzip it into
  `Temporal Change Results/Temporal_Change_Results/` before running `evaluation.py`.
- Every stage's outputs are already committed, so each stage can be rerun or
  inspected on its own.

### Preprocessing details

```sh
python preprocessing.py --limit 1      # trial run on one observation
python preprocessing.py                # full dataset
python -m unittest discover -s tests -v
python tests/verify_delivered_dataset.py
```

The first command checks all inputs and processes one observation as a trial.
The second processes the full dataset and replaces the trial report. Existing
outputs are reused only after their provenance, values, masks and
georeferencing pass verification; changed inputs or code trigger regeneration.
A nonzero exit status means processing or inventory errors: see
`Processed Dataset/validation_issues.csv` and the per-image report.

The source folder defaults to `RawDataset`; pass `--source relative/folder` to use
another. Outputs always go to `Processed Dataset` next to the script. Optional
display-only smoothing is available with `--preview-smoothing gaussian` (or
`median`); it never changes the GeoTIFF exports. The full run needs about 0.5 GB
of extra disk space. See the [Processed Dataset README](<Processed Dataset/README.md>)
for mask definitions, file formats and limitations.

## Repository structure

```
RawDataset/                   Original Sentinel-2 chips, PNG previews, metadata.csv
Processed Dataset/            Reflectance GeoTIFFs, valid-pixel masks, reports
Segmentation/Results_BSI_Otsu/  BSI + Otsu masks and summary
Texture Morphology Results/   Refined masks (masks/<year>/), previews, summary.csv
Temporal Change Results/      Change maps, area tables and figures (zipped)
Evaluation Results/           Agreement metrics, comparison tables, figures
tests/                        Preprocessing tests and dataset verification
```

## Limitations and future work

- No hand-labelled reference masks: accuracy was assessed visually and by
  agreement with the BSI + Otsu baseline. Labelling a sample of images would allow
  true IoU/Dice evaluation.
- At 10 m resolution, mixed pixels and narrow roads are hard to resolve, and
  the method does not separate dunes from flat sand sheets.
- Possible next steps: machine learning (Random Forest, SVM) or deep learning
  (U-Net) segmentation trained on the refined masks; Siamese or ConvLSTM networks
  for change detection and forecasting; higher-resolution imagery with sub-pixel
  correlation (COSI-Corr) to measure dune movement.

## Data source

Contains modified Copernicus Sentinel data (2019–2026), processed by the project
team. Sentinel-2 imagery is provided by the European Union's Copernicus Programme
and the European Space Agency.
