"""Dune segmentation with texture analysis and morphological refinement.

Steps for every image in the processed dataset:
    1. Load reflectance and the valid-pixel mask.
    2. Compute features: colour (brightness, redness, NDVI, bare soil index) + texture (contrast, entropy).
    3. Group pixels into clusters with k-means (one model for all images).
    4. Call a cluster "dune" if it looks like sand and is not extremely textured,
       then remove salt/gypsum flats (sabkha) using the SWIR ratio B11/B12.
    5. Clean the dune mask with morphology (opening, closing, fill holes, drop specks).
    6. Save the mask, a preview image and a summary table.

All settings live in Config. Change them there (or pass your own Config to run()).
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from scipy.cluster.vq import kmeans2, vq


# %% Settings
@dataclass
class Config:
    # Folders
    input_dir: Path = Path("Processed Dataset")
    output_dir: Path = Path("Segmentation Results")
    locations: tuple = ()            # e.g. ("L1", "L10"); empty = all sites

    # Band order in the processed GeoTIFFs
    band_order: tuple = ("B02", "B03", "B04", "B08", "B11", "B12")

    # Texture
    window: int = 15                 # texture window in pixels (15 px = 150 m)
    grey_levels: int = 32            # brightness is binned into this many levels
    max_reflectance: float = 0.6     # brightness range used for binning

    # Clustering
    n_clusters: int = 6
    pixels_per_image: int = 3000     # random pixels per image used to train k-means
    outlier_limit: float = 5.0       # cap standardized features at +/- this
    seed: int = 0

    # Dune rule (in standard deviations from the dataset average)
    min_sand_score: float = 0.0      # sand score = brightness + redness - NDVI
    max_contrast: float = 2.0        # above this, texture is buildings/roads, not dunes
    max_swir_ratio: float = 1.15     # B11/B12 above this = salt/gypsum flat (sabkha), not sand

    # Morphology (sizes in pixels; 1 px = 10 m x 10 m)
    opening_radius: int = 1          # removes thin specks and lines
    closing_radius: int = 2          # bridges small gaps
    max_hole_size: int = 50          # fill enclosed gaps up to this many pixels
    min_dune_size: int = 50          # drop dune patches smaller than this


# %% Step 1: load one image
def load_image(cfg, row):
    with rasterio.open(cfg.input_dir / row.processed_filename) as src:
        reflectance = src.read().astype(np.float32)
        profile = src.profile
    with rasterio.open(cfg.input_dir / row.mask_filename) as src:
        valid = src.read(1).astype(bool)
    valid &= np.isfinite(reflectance).all(axis=0)
    bands = dict(zip(cfg.band_order, reflectance))
    return bands, valid, profile


# %% Step 2: features
def window_mean(image, valid, size):
    """Average of `image` in a size x size window, ignoring invalid pixels."""
    weights = valid.astype(np.float32)
    total = ndi.uniform_filter(np.where(valid, image, 0) * weights, size)
    count = ndi.uniform_filter(weights, size)
    return total / np.maximum(count, 1e-6)


def glcm_contrast(levels, valid, size):
    """GLCM contrast: average squared grey-level difference between neighbouring
    pixels (right, down, and both diagonals) inside the window.
    Smooth sand -> low; ripples -> medium; buildings and roads -> high."""
    h, w = levels.shape
    padded_levels = np.pad(levels, 1, mode="edge")
    padded_valid = np.pad(valid, 1, constant_values=False)
    contrast = np.zeros(levels.shape, np.float32)
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        neighbour = padded_levels[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        neighbour_valid = padded_valid[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        diff_squared = (levels - neighbour).astype(np.float32) ** 2
        contrast += window_mean(diff_squared, valid & neighbour_valid, size)
    return np.log1p(contrast / 4)


def local_entropy(levels, valid, size, n_levels):
    """How mixed the grey levels are in the window (0 = all the same)."""
    entropy = np.zeros(levels.shape, np.float32)
    for level in range(n_levels):
        p = np.clip(window_mean((levels == level).astype(np.float32), valid, size), 0, 1)
        entropy -= p * np.log2(np.where(p > 0, p, 1))
    return entropy


def compute_features(cfg, bands, valid):
    """Returns a dict of 2-D feature images (NaN where invalid)."""
    eps = 1e-6
    brightness = (bands["B02"] + bands["B03"] + bands["B04"]) / 3
    redness = (bands["B04"] - bands["B02"]) / (bands["B04"] + bands["B02"] + eps)
    ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + eps)
    soil = (bands["B11"] + bands["B04"]) - (bands["B08"] + bands["B02"])
    bare_soil_index = soil / ((bands["B11"] + bands["B04"]) + (bands["B08"] + bands["B02"]) + eps)

    binned = np.clip(np.nan_to_num(brightness) / cfg.max_reflectance, 0, 0.999)
    levels = (binned * cfg.grey_levels).astype(np.int16)

    features = {
        "brightness": brightness,
        "redness": redness,
        "ndvi": ndvi,
        "bare_soil_index": bare_soil_index,   # separates sand from salt/gypsum flats
        "contrast": glcm_contrast(levels, valid, cfg.window),
        "entropy": local_entropy(levels, valid, cfg.window, cfg.grey_levels),
    }
    for name in features:
        features[name] = np.where(valid, features[name], np.nan)
    return features


def to_table(features, valid):
    """Feature images -> (n_valid_pixels, n_features) table."""
    return np.column_stack([image[valid] for image in features.values()])


# %% Step 3: clustering
def standardize(table, mean, std, limit):
    """Put every feature on the same scale (0 = average, 1 = one standard deviation).
    Values are capped at +/- limit so a few extreme pixels (water, metal roofs)
    cannot claim a whole cluster for themselves."""
    return np.clip((table - mean) / std, -limit, limit)


def train_kmeans(cfg, samples):
    mean, std = samples.mean(axis=0), samples.std(axis=0) + 1e-6
    z = standardize(samples, mean, std, cfg.outlier_limit)
    centroids, _ = kmeans2(z, cfg.n_clusters, minit="++", seed=cfg.seed)
    return {"mean": mean, "std": std, "centroids": centroids}


def predict_clusters(cfg, model, table):
    z = standardize(table, model["mean"], model["std"], cfg.outlier_limit)
    labels, _ = vq(z, model["centroids"])
    return labels


# %% Step 4: which clusters are dunes?
def pick_dune_clusters(cfg, model, feature_names):
    """Centroids are in standard deviations, so 0 = dataset average."""
    c = dict(zip(feature_names, model["centroids"].T))
    sand_score = c["brightness"] + c["redness"] - c["ndvi"]
    is_dune = (sand_score > cfg.min_sand_score) & (c["contrast"] < cfg.max_contrast)
    summary = pd.DataFrame({"cluster": range(cfg.n_clusters), "sand_score": sand_score.round(2),
                            "contrast": c["contrast"].round(2), "is_dune": is_dune})
    return set(np.flatnonzero(is_dune)), summary


def salt_flat_mask(cfg, bands):
    """Salt and gypsum flats (sabkha) can look like pale sand in colour, but these
    minerals absorb short-wave infrared near 2.2 um (band B12) and quartz sand does
    not. Dune sand has B11/B12 of about 1.04-1.10; salt flats reach 1.2-1.4."""
    return bands["B11"] / (bands["B12"] + 1e-6) > cfg.max_swir_ratio


# %% Step 5: morphological refinement
def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x**2 + y**2 <= radius**2


def refine(cfg, mask, valid):
    pad = max(cfg.opening_radius, cfg.closing_radius)
    m = np.pad(mask, pad, mode="edge")          # padding stops the image border from eroding
    if cfg.opening_radius:
        m = ndi.binary_opening(m, disk(cfg.opening_radius))
    if cfg.closing_radius:
        m = ndi.binary_closing(m, disk(cfg.closing_radius))
    m = m[pad:-pad, pad:-pad] if pad else m

    # Fill small enclosed gaps (large ones are real interdune flats).
    gaps, _ = ndi.label(~m & valid)
    gap_sizes = np.bincount(gaps.ravel())
    touches_edge = np.unique(np.r_[gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]])
    fill = gap_sizes <= cfg.max_hole_size
    fill[[0, *touches_edge]] = False
    m |= fill[gaps]

    # Drop tiny dune patches.
    patches, _ = ndi.label(m)
    keep = np.bincount(patches.ravel()) >= cfg.min_dune_size
    keep[0] = False
    return keep[patches] & valid


# %% Step 6: save outputs
def save_mask(path, mask, valid, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(valid, mask, 255).astype(np.uint8)   # 1 dune, 0 not dune, 255 invalid
    meta = {**profile, "count": 1, "dtype": "uint8", "nodata": 255, "compress": "deflate"}
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(out, 1)


def save_preview(cfg, path, bands, features, clusters, raw, refined):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]]) / cfg.max_reflectance
    rgb = np.nan_to_num(np.clip(rgb, 0, 1))
    overlay = rgb * 0.5
    overlay[refined] = overlay[refined] * 0.4 + np.array([1.0, 0.85, 0.0]) * 0.6   # dune = yellow
    panels = [(rgb, "RGB", None), (features["contrast"], "Texture: GLCM contrast", "magma"),
              (clusters, "k-means clusters", "tab10"), (raw, "Dune mask before refinement", "gray"),
              (overlay, "After refinement (yellow = dune)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    for ax, (image, title, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(path.stem)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# %% Run everything
def run(cfg=Config()):
    records = pd.read_csv(cfg.input_dir / "preprocessing_report.csv")
    records = records[records.status.isin(["exported", "reused"])]
    if cfg.locations:
        records = records[records.location_id.isin(cfg.locations)]
    rng = np.random.default_rng(cfg.seed)

    # Pass 1: collect random pixels from every image and train one k-means model.
    samples = []
    for row in records.itertuples():
        bands, valid, _ = load_image(cfg, row)
        features = compute_features(cfg, bands, valid)
        table = to_table(features, valid)
        pick = rng.choice(len(table), min(cfg.pixels_per_image, len(table)), replace=False)
        samples.append(table[pick])
    feature_names = list(features)
    model = train_kmeans(cfg, np.concatenate(samples))
    dune_clusters, cluster_table = pick_dune_clusters(cfg, model, feature_names)
    print(cluster_table.to_string(index=False))

    # Pass 2: segment and refine every image.
    results = []
    for row in records.itertuples():
        bands, valid, profile = load_image(cfg, row)
        features = compute_features(cfg, bands, valid)

        clusters = np.full(valid.shape, np.nan)
        clusters[valid] = predict_clusters(cfg, model, to_table(features, valid))
        raw = np.isin(clusters, list(dune_clusters)) & ~salt_flat_mask(cfg, bands)
        refined = refine(cfg, raw, valid)

        name = f"{row.location}_{row.year}"
        save_mask(cfg.output_dir / "masks" / str(row.year) / f"{name}_dune.tif", refined, valid, profile)
        save_preview(cfg, cfg.output_dir / "previews" / f"{name}.png", bands, features, clusters, raw, refined)

        pixel_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
        results.append({"location": row.location, "year": row.year,
                        "dune_fraction": round(refined.sum() / valid.sum(), 4),
                        "dune_area_km2": round(refined.sum() * pixel_km2, 3),
                        "dune_patches": ndi.label(refined)[1]})
        print(f"{name}: {raw.mean():.1%} dune before refinement, {refined.mean():.1%} after")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_table.to_csv(cfg.output_dir / "clusters.csv", index=False)
    pd.DataFrame(results).to_csv(cfg.output_dir / "summary.csv", index=False)
    print(f"Saved results to {cfg.output_dir}")


if __name__ == "__main__":
    run()