"""
Steps for every image:
    1. Load reflectance, the valid-pixel mask and the BSI-Otsu sand mask.
    2. Texture analysis: GLCM contrast and local entropy in a sliding window.
    3. Remove sand pixels that are too textured (roads, buildings, farms) and
       salt/gypsum flats (sabkha) detected with the SWIR ratio B11/B12.
    4. Clean the mask with morphology (opening, closing, fill holes, drop specks).
    5. Save the refined mask, a preview image and a summary table.

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


# %% Settings
@dataclass
class Config:
    # Folders
    input_dir: Path = Path("Processed Dataset")
    segmentation_dir: Path = Path("Segmentation/Results_BSI_Otsu")
    segmentation_suffix: str = "_BSI_Otsu_mask.tif"
    output_dir: Path = Path("Texture Morphology Results")
    locations: tuple = ()            # e.g. ("L1", "L10"); empty = all sites

    # Band order in the processed GeoTIFFs
    band_order: tuple = ("B02", "B03", "B04", "B08", "B11", "B12")

    # Texture
    window: int = 15                 # texture window in pixels (15 px = 150 m)
    grey_levels: int = 32            # brightness is binned into this many levels
    max_reflectance: float = 0.6     # brightness range used for binning
    pixels_per_image: int = 3000     # random pixels per image used for dataset statistics
    seed: int = 0

    # Refinement rules
    max_texture: float = 2.0         # texture score (std. deviations above the dataset average)
    max_swir_ratio: float = 1.15     # B11/B12 above this = salt/gypsum flat (sabkha), not sand

    # Morphology (sizes in pixels; 1 px = 10 m x 10 m)
    opening_radius: int = 1          # removes thin specks and lines
    closing_radius: int = 2          # bridges small gaps
    max_hole_size: int = 50          # fill enclosed gaps up to this many pixels
    min_dune_size: int = 50          # drop sand patches smaller than this


# %% Step 1: load one image and its segmentation mask
def load_image(cfg, row):
    with rasterio.open(cfg.input_dir / row.processed_filename) as src:
        reflectance = src.read().astype(np.float32)
        profile = src.profile
    with rasterio.open(cfg.input_dir / row.mask_filename) as src:
        valid = src.read(1).astype(bool)
    valid &= np.isfinite(reflectance).all(axis=0)
    bands = dict(zip(cfg.band_order, reflectance))
    return bands, valid, profile


def load_segmentation(cfg, row, valid):
    """Sand mask from the segmentation step: 1 sand, 0 not sand, 255 invalid."""
    path = cfg.segmentation_dir / f"{row.location}_{row.year}{cfg.segmentation_suffix}"
    with rasterio.open(path) as src:
        mask = src.read(1)
    valid &= mask != 255
    return (mask == 1) & valid, valid


# %% Step 2: texture analysis
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


def compute_texture(cfg, bands, valid):
    """Returns {"contrast": image, "entropy": image} (NaN where invalid)."""
    brightness = (bands["B02"] + bands["B03"] + bands["B04"]) / 3
    binned = np.clip(np.nan_to_num(brightness) / cfg.max_reflectance, 0, 0.999)
    levels = (binned * cfg.grey_levels).astype(np.int16)
    texture = {
        "contrast": glcm_contrast(levels, valid, cfg.window),
        "entropy": local_entropy(levels, valid, cfg.window, cfg.grey_levels),
    }
    return {name: np.where(valid, image, np.nan) for name, image in texture.items()}


def texture_score(texture, stats):
    """Average of contrast and entropy, each in standard deviations from the
    dataset average (0 = typical pixel, 2 = much rougher than usual)."""
    z = [(texture[name] - stats[name][0]) / stats[name][1] for name in texture]
    return sum(z) / len(z)


# %% Step 3: refinement rules
def salt_flat_mask(cfg, bands):
    """Salt and gypsum flats (sabkha) can look like pale sand in colour, but these
    minerals absorb short-wave infrared near 2.2 um (band B12) and quartz sand does
    not. Dune sand has B11/B12 of about 1.04-1.10; salt flats reach 1.2-1.4."""
    return bands["B11"] / (bands["B12"] + 1e-6) > cfg.max_swir_ratio


# %% Step 4: morphological refinement
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

    # Drop tiny sand patches.
    patches, _ = ndi.label(m)
    keep = np.bincount(patches.ravel()) >= cfg.min_dune_size
    keep[0] = False
    return keep[patches] & valid


# %% Step 5: save outputs
def save_mask(path, mask, valid, profile):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(valid, mask, 255).astype(np.uint8)   # 1 sand, 0 not sand, 255 invalid
    meta = {**profile, "count": 1, "dtype": "uint8", "nodata": 255, "compress": "deflate"}
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(out, 1)


def save_preview(cfg, path, bands, score, segmented, cleaned, refined):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]]) / cfg.max_reflectance
    rgb = np.nan_to_num(np.clip(rgb, 0, 1))
    overlay = rgb * 0.5
    overlay[refined] = overlay[refined] * 0.4 + np.array([1.0, 0.85, 0.0]) * 0.6   # sand = yellow
    panels = [(rgb, "RGB", None), (score, "Texture score", "magma"),
              (segmented, "Segmentation (BSI + Otsu)", "gray"),
              (cleaned, "After texture + salt-flat rules", "gray"),
              (overlay, "After morphology (yellow = sand)", None)]
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

    # Pass 1: dataset-wide texture statistics, so the same rule applies to every year.
    samples = {"contrast": [], "entropy": []}
    for row in records.itertuples():
        bands, valid, _ = load_image(cfg, row)
        texture = compute_texture(cfg, bands, valid)
        pick = rng.choice(valid.sum(), min(cfg.pixels_per_image, valid.sum()), replace=False)
        for name in samples:
            samples[name].append(texture[name][valid][pick])
    stats = {name: (np.mean(v), np.std(v) + 1e-6) for name, v in
             ((name, np.concatenate(parts)) for name, parts in samples.items())}

    # Pass 2: refine the segmentation mask of every image.
    results = []
    for row in records.itertuples():
        bands, valid, profile = load_image(cfg, row)
        segmented, valid = load_segmentation(cfg, row, valid)
        texture = compute_texture(cfg, bands, valid)
        score = texture_score(texture, stats)

        too_rough = segmented & (score > cfg.max_texture)
        salt_flat = segmented & salt_flat_mask(cfg, bands)
        cleaned = segmented & ~too_rough & ~salt_flat
        refined = refine(cfg, cleaned, valid)

        name = f"{row.location}_{row.year}"
        save_mask(cfg.output_dir / "masks" / str(row.year) / f"{name}_refined.tif", refined, valid, profile)
        save_preview(cfg, cfg.output_dir / "previews" / f"{name}.png", bands, score, segmented, cleaned, refined)

        n = valid.sum()
        pixel_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
        results.append({"location": row.location, "year": row.year,
                        "segmented_fraction": round(segmented.sum() / n, 4),
                        "removed_texture": round(too_rough.sum() / n, 4),
                        "removed_salt_flat": round((salt_flat & ~too_rough).sum() / n, 4),
                        "refined_fraction": round(refined.sum() / n, 4),
                        "refined_area_km2": round(refined.sum() * pixel_km2, 3),
                        "patches_before": ndi.label(segmented)[1],
                        "patches_after": ndi.label(refined)[1]})
        print(f"{name}: {segmented.sum() / n:.1%} sand from segmentation, {refined.sum() / n:.1%} after refinement")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(cfg.output_dir / "summary.csv", index=False)
    print(f"Saved results to {cfg.output_dir}")


if __name__ == "__main__":
    run()