import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from tqdm import tqdm

# ----------------------------------
# Path configuration
# ----------------------------------
base_dir = r"G:\FIRE_NS\settlement"

towns_dir  = os.path.join(base_dir, "GCTB", "Towns_2000_2022")
cities_dir = os.path.join(base_dir, "GCTB", "Cities_2000_2022")
gurs_dir   = os.path.join(base_dir, "GURS")

out_dir = os.path.join(base_dir, "GCTB", "Dominant")
os.makedirs(out_dir, exist_ok=True)

years = [2005, 2010, 2015, 2020]

# ----------------------------------
# Main loop
# ----------------------------------
for year in tqdm(years, desc="Processing years"):

    towns_shp   = os.path.join(towns_dir,  f"Towns_{year}.shp")
    cities_shp  = os.path.join(cities_dir, f"Cities_{year}.shp")
    template_tif = os.path.join(gurs_dir,  f"GURS_GBLU_{year}.tif")

    out_tif = os.path.join(out_dir, f"GCTB_dominate_{year}.tif")

    # ----------------------------------
    # Load template raster
    # ----------------------------------
    with rasterio.open(template_tif) as src:
        meta = src.meta.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs

    # ----------------------------------
    # Load and reproject the vector layer
    # ----------------------------------
    gdf_towns  = gpd.read_file(towns_shp).to_crs(crs)
    gdf_cities = gpd.read_file(cities_shp).to_crs(crs)

    # ----------------------------------
    # Rasterization
    # ----------------------------------
    towns_raster = rasterize(
        [(geom, 1) for geom in tqdm(
            gdf_towns.geometry,
            desc=f"{year} rasterizing towns",
            leave=False
        )],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    cities_raster = rasterize(
        [(geom, 1) for geom in tqdm(
            gdf_cities.geometry,
            desc=f"{year} rasterizing cities",
            leave=False
        )],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # ----------------------------------
    # Dominant type identification
    # ----------------------------------
    out = np.full(shape, np.nan, dtype='float32')

    # Only Towns
    out[(towns_raster == 1) & (cities_raster == 0)] = 1

    # Only Cities
    out[(towns_raster == 0) & (cities_raster == 1)] = 2

    # Simultaneous overlap (default: Towns have priority; can be set to 2 if required)
    out[(towns_raster == 1) & (cities_raster == 1)] = 1

    # ----------------------------------
    # Save as a GeoTIFF file
    # ----------------------------------
    meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': np.nan,
        'compress': 'lzw'
    })

    with rasterio.open(out_tif, 'w', **meta) as dst:
        dst.write(out, 1)

# ----------------------------------
print("✔ All years processed successfully.")
print("Output directory:")
print(out_dir)
