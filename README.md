# Sand Dune Segmentation and Temporal Change Detection Along the Abqaiq–Salasil Highway Using Satellite Image Processing.

Dataset: [RawDataset](RawDataset/).

## Data loading and preprocessing

The pipeline in [preprocessing.py](preprocessing.py) validates all source GeoTIFFs
against `RawDataset/metadata.csv`, checks each location's temporal grid, scales
six spectral bands to Float32 reflectance, and masks invalid pixels using SCL
and the source raster masks. All generated files go into
[Processed Dataset](<Processed Dataset/>). The original dataset is read-only.

### Run in VS Code

Open this repository folder in VS Code and open its integrated terminal. Use
Python 3.12 or newer (the delivered run used Python 3.14). Create an environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `python -m venv .venv` and `.venv\Scripts\Activate.ps1`.
In VS Code, run **Python: Select Interpreter** and select `.venv`. Then run:

```sh
python preprocessing.py --limit 1
python preprocessing.py
python -m unittest discover -s tests -v
python tests/verify_delivered_dataset.py
```

The first command preflights all inputs and processes one observation for a trial.
The second processes the complete dataset and replaces the trial report with the
full report. Existing pipeline outputs are reused only after their provenance,
numerical values, mask values and georeferencing pass verification. Changed inputs
or code trigger regeneration. Files not marked as belonging to this pipeline
are never overwritten. A nonzero exit status indicates processing/inventory errors;
inspect `Processed Dataset/validation_issues.csv` and the per-image report.

The source folder defaults to `RawDataset` (the current name); a legacy `dataset`
folder is also supported. To use another source, pass `--source relative/folder`.
Paths in metadata remain relative to that source folder. Outputs are always placed
in `Processed Dataset` next to the script, even when launched from another directory.

Optional display-only smoothing is available with
`python preprocessing.py --preview-smoothing gaussian` (or `median`). It is disabled
by default and never changes the scientific GeoTIFF exports. The complete dataset
requires approximately 0.5 GB of additional disk space, depending on compression.

See [output documentation](<Processed Dataset/README.md>) for mask definitions,
file formats, scientific limitations and examples for teammates, and
[the paper methods summary](PREPROCESSING_METHODS.md) for Section IV material.
