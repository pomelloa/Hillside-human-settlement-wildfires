import os
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==================== Path configuration ====================
base_dir = r"G:\FIRE\GFA"
shp_dir = os.path.join(base_dir, "SHP_perimeters")
mcd_dir = os.path.join(base_dir, "MCD12Q1")
mod14_dir = os.path.join(base_dir, "AFD")
ndvi_dir = os.path.join(base_dir, "NDVI")
evi_dir = os.path.join(base_dir, "EVI")
out_dir = os.path.join(base_dir, "output_mask")
os.makedirs(out_dir, exist_ok=True)

os.environ["PROJ_LIB"] = r"D:\miniconda3\envs\gmdl\Library\share\proj"

YEARS = [2005, 2010, 2015, 2020]
N_PROC = min(cpu_count() - 2, 16)


# ==================== Main function====================

def compute_polygon_stats(geom, mcd_data, mod14_data, ndvi_data, evi_data, transform):
    """计算单个polygon的统计值"""
    from rasterio import features

    mask = features.geometry_mask([geom],
                                  out_shape=mcd_data.shape,
                                  transform=transform,
                                  invert=True)

    results = {}
    try:
        # Extract valid pixels
        mcd_valid = mcd_data[mask]
        mod_valid = mod14_data[mask]
        ndvi_valid = ndvi_data[mask]
        evi_valid = evi_data[mask]

        # Category-wise statistics
        if mcd_valid.size > 0:
            crop_ratio = np.sum(mcd_valid == 12) / mcd_valid.size
            urban_ratio = np.sum(mcd_valid == 13) / mcd_valid.size
        else:
            crop_ratio = urban_ratio = 0

        results["crop_ratio"] = crop_ratio
        results["urban_ratio"] = urban_ratio
        results["fire_days_mean"] = np.nanmean(mod_valid[mod_valid > 0]) if np.any(mod_valid > 0) else 0
        results["NDVI_mean"] = np.nanmean(ndvi_valid[ndvi_valid > 0]) if np.any(ndvi_valid > 0) else 0
        results["EVI_mean"] = np.nanmean(evi_valid[evi_valid > 0]) if np.any(evi_valid > 0) else 0

    except Exception:
        results = {"crop_ratio": 0, "urban_ratio": 0, "fire_days_mean": 0, "NDVI_mean": 0, "EVI_mean": 0}

    return results


def process_block(block_args):
    """Process all polygon features within one block"""
    block_id, gdf_block, paths, transform = block_args
    mcd_path, mod14_path, ndvi_path, evi_path = paths

    # Read the raster data for the corresponding window
    with rasterio.open(mcd_path) as mcd, \
            rasterio.open(mod14_path) as mod, \
            rasterio.open(ndvi_path) as ndvi, \
            rasterio.open(evi_path) as evi:
        bounds = gdf_block.total_bounds
        window = rasterio.windows.from_bounds(*bounds, transform=mcd.transform, width=mcd.width, height=mcd.height)
        window = window.round_offsets().round_lengths()

        # Load data
        mcd_data = mcd.read(1, window=window)
        mod_data = mod.read(1, window=window)
        ndvi_data = ndvi.read(1, window=window)
        evi_data = evi.read(1, window=window)

        # Window-level transform
        transform_win = mcd.window_transform(window)

        # Compute for each polygon
        results = []
        for geom in gdf_block.geometry:
            r = compute_polygon_stats(geom, mcd_data, mod_data, ndvi_data, evi_data, transform_win)
            results.append(r)

    df_block = pd.DataFrame(results)
    return pd.concat([gdf_block.reset_index(drop=True), df_block], axis=1)


# ==================== Main ====================
for year in YEARS:
    shp_path = os.path.join(shp_dir, f"GFA_v20240409_perimeters_{year}.shp")
    mcd_path = os.path.join(mcd_dir, f"MCD12Q1_{year}.tif")
    mod14_path = os.path.join(mod14_dir, f"MOD14A1_AFD_{year}.tif")
    ndvi_path = os.path.join(ndvi_dir, f"MOD13A1_mean_ndvi_{year}.tif")
    evi_path = os.path.join(evi_dir, f"MOD13A1_mean_evi__{year}.tif")
    out_path = os.path.join(out_dir, f"GFA_clean_{year}.shp")

    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    gdf["geometry"] = gdf["geometry"].simplify(0.001, preserve_topology=True)

    # ==== Build a spatial index and split into blocks ====
    xmin, ymin, xmax, ymax = gdf.total_bounds
    block_size = 2.0
    x_blocks = np.arange(xmin, xmax, block_size)
    y_blocks = np.arange(ymin, ymax, block_size)
    blocks = []
    for x in x_blocks:
        for y in y_blocks:
            blocks.append(box(x, y, x + block_size, y + block_size))

    tree = STRtree(gdf.geometry)
    tasks = []
    for i, blk in enumerate(blocks):
        matches = tree.query(blk)
        if not matches:
            continue
        sub_gdf = gdf.iloc[matches].copy()
        tasks.append((i, sub_gdf, (mcd_path, mod14_path, ndvi_path, evi_path), None))

    with Pool(processes=N_PROC) as pool:
        parts = list(tqdm(pool.imap(process_block, tasks), total=len(tasks), desc=f"{year} blocks"))

    gdf_all = pd.concat(parts, ignore_index=True)

    # ==== Filter ====
    cond = (
            (gdf_all["urban_ratio"] <= 0.4) &
            (gdf_all["crop_ratio"] <= 0.4) &
            (gdf_all["fire_days_mean"] >= 0.1) &
            (gdf_all["fire_days_mean"] <= 5) &
            (gdf_all["NDVI_mean"] >= 0.08) &
            (gdf_all["EVI_mean"] >= 0.08)
    )
    gdf_filtered = gdf_all[cond].copy()
    gdf_filtered.to_file(out_path, encoding="utf-8")

    print(f"✅ {year}: {len(gdf_filtered)} / {len(gdf_all)} polygons kept")

print("🎯 All years processed successfully with block-wise optimization.")
