from osgeo import gdal
import numpy as np
from tqdm import tqdm
import os

# -------------------------------
# Time parameters and path templates
# -------------------------------
years = [2005, 2010, 2015, 2020]

gblu_template = r"D:\GURS\gblu_resam_clip_fixed.tif"
gurs_template = r"D:\GURS\ghs_{year}_resam_align_reclass.tif"
output_template = r"D:\GURS\ghs_gblu_r100_{year}_align.tif"

# -------------------------------
# Loop through each year
# -------------------------------
for year in years:
    gblu_path = gblu_template.format(year=year)
    gurs_path = gurs_template.format(year=year)
    output_path = output_template.format(year=year)

    print(f"Processing year {year}...")

    # Read data
    src_gblu = gdal.Open(gblu_path, gdal.GA_ReadOnly)
    src_gurs = gdal.Open(gurs_path, gdal.GA_ReadOnly)

    if src_gblu is None or src_gurs is None:
        print(f"Read failed：{gblu_path} or {gurs_path}")
        continue

    gblu_band = src_gblu.GetRasterBand(1)
    gurs_band = src_gurs.GetRasterBand(1)

    width, height = src_gblu.RasterXSize, src_gblu.RasterYSize
    nodata_gblu = gblu_band.GetNoDataValue()
    nodata_gurs = gurs_band.GetNoDataValue()

    # Consistency check
    if (width != src_gurs.RasterXSize) or (height != src_gurs.RasterYSize):
        print(f"{year} Image dimensions mismatch; skipping")
        continue

    # Generate output
    if os.path.exists(output_path):
        os.remove(output_path)

    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        output_path, width, height, 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE"]
    )
    dst_ds.SetGeoTransform(src_gblu.GetGeoTransform())
    dst_ds.SetProjection(src_gblu.GetProjection())

    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(0)

    # Process in blocks
    # Valid classes: 11, 12, 21, 22, 31, 32
    # -> the ones digit indicates terrain (1 = plain; 2/3 = highland), and the tens digit indicates settlement type (1 = urban; 2 = rural).
    block_w, block_h = 1024, 1024
    valid_values = np.array([11, 12, 21, 22, 31, 32], dtype=np.uint8)

    with tqdm(total=((width + block_w - 1) // block_w) * ((height + block_h - 1) // block_h),
              desc=f"Year {year}", unit="block") as pbar:
        for y in range(0, height, block_h):
            for x in range(0, width, block_w):
                w = min(block_w, width - x)
                h = min(block_h, height - y)

                gblu_block = gblu_band.ReadAsArray(x, y, w, h).astype(np.uint8)
                ghs_block = gurs_band.ReadAsArray(x, y, w, h).astype(np.uint8)

                # Replace NoData values with 0
                if nodata_gblu is not None:
                    gblu_block[gblu_block == nodata_gblu] = 0
                if nodata_gurs is not None:
                    ghs_block[ghs_block == nodata_gurs] = 0

                result = 10 * gblu_block + ghs_block
                mask = np.isin(result, valid_values)
                result[~mask] = 0
                result = result.astype(np.uint8)

                dst_band.WriteArray(result, x, y)
                pbar.update(1)

    dst_band.FlushCache()
    dst_ds = None
    src_gblu = None
    src_gurs = None

    print(f"Processing for year {year} completed. Results saved to: {output_path}")
