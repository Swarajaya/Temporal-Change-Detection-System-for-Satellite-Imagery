import rasterio
import os
import sys

# ---- Get year from command line ----
year = sys.argv[1]   # e.g. 2020
base_path = f"data/raw/{year}"

# ---- Band paths ----
blue_path  = os.path.join(base_path, "B02.jp2")  # Blue
green_path = os.path.join(base_path, "B03.jp2")  # Green
red_path   = os.path.join(base_path, "B04.jp2")  # Red

output_path = os.path.join(base_path, "image.tif")

# ---- Read bands & create RGB ----
with rasterio.open(red_path) as red, \
     rasterio.open(green_path) as green, \
     rasterio.open(blue_path) as blue:

    meta = red.meta
    meta.update(count=3)

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(red.read(1), 1)    # Red
        dst.write(green.read(1), 2)  # Green
        dst.write(blue.read(1), 3)   # Blue

print(f"RGB image created for {year}: image.tif")
