"""Export predictions as GeoTIFF files."""
import numpy as np
import rasterio


def export_geotiff(pred_s, pred_w, pred_d, pred_gvvi, grid_df, raster_meta, output_dir):
    """Write predictions into 247x245 GeoTIFF rasters matching official format."""
    print("=" * 60)
    print("EXPORT: GeoTIFF Prediction Maps")
    print("=" * 60)

    height = raster_meta["height"]
    width = raster_meta["width"]
    transform = raster_meta["transform"]
    crs = raster_meta["crs"]

    xs = grid_df["grid_idx_x"].values
    ys = grid_df["grid_idx_y"].values

    def write_tif(data, path):
        raster = np.zeros((height, width), dtype=np.float32)
        raster[xs, ys] = data
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype=raster.dtype, transform=transform, crs=crs) as dst:
            dst.write(raster, 1)
        print(f"  {path}: shape={raster.shape}, nonzero={(raster!=0).sum()}")

    write_tif(pred_s.astype(np.float32), f"{output_dir}/pred_GVVI_S.tif")
    write_tif(pred_w.astype(np.float32), f"{output_dir}/pred_GVVI_W.tif")
    write_tif(pred_d.astype(np.float32), f"{output_dir}/pred_GVVI_D.tif")
    write_tif(pred_gvvi.astype(np.float32), f"{output_dir}/pred_GVVI.tif")

    print("GeoTIFF export complete.\n")
