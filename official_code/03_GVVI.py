import pandas as pd
import os
import rasterio
import numpy as np
from utils import normalize_raster
import time

name_dict = {
    'SVI': 'GVVI_S',
    'WVI': 'GVVI_W',
    'DVI': 'GVVI_D'
}

weight_dict = {
    'SVI': 3,
    'WVI': 2,
    'DVI': 1
}

raster_path = 'data/data/veg_raster/veg_raster_buffer_100.tif'

def write_raster(data, save_path, height, width, transform, crs):
    with rasterio.open(save_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, transform=transform, crs=crs) as dst:
            dst.write(data, 1)

def grid_result():

    df_path = 'data/GVVI/temp/'

    df_list = ['SVI', 'WVI', 'DVI']

    with rasterio.open(raster_path) as src:
        raster = src.read(1)

        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

        for df_name in df_list:
            df = pd.read_csv(df_path+df_name+'.csv')

            grid = np.zeros((height,width))

            for i in range(len(df)):
                x = df.loc[i,'grid_idx_x']
                y = df.loc[i,'grid_idx_y']
                value = raster[x,y]
                if value != 0:
                    grid[x,y] += df.loc[i,'total_pixel']

            write_raster(grid, f'data/GVVI/total_pixel/{name_dict[df_name]}.tif', height, width, transform, crs)
    print('save raster')

def GVVI():

    GVVI = []
    total_weight = weight_dict['SVI'] + weight_dict['WVI'] + weight_dict['DVI']

    with rasterio.open(raster_path) as src:
        mask_raster = src.read(1)
        mask = mask_raster != 0

    for d in ['SVI', 'WVI', 'DVI']:
        GVVI_raster_path = f'data/GVVI/total_pixel/{name_dict[d]}.tif'
        with rasterio.open(GVVI_raster_path) as src:
            raster = src.read(1)
            print(f'processing {d} raster:', 'max value:',np.max(raster), 'min value:', np.min(raster))
            raster = normalize_raster(raster)
            height = src.height
            width = src.width
            transform = src.transform
            crs = src.crs

            weight = weight_dict[d]/total_weight
            print(f'weight for {d}:', weight)
            raster = raster * weight
            GVVI.append(raster)

    GVVI = np.sum(GVVI, axis=0).reshape((height, width))
    print('--------------------')
    print('GVVI max value:',np.max(GVVI[mask]), 'min value:', np.min(GVVI[mask]), 'mean value:', np.mean(GVVI[mask]))
    write_raster(GVVI, 'data/GVVI/GVVI.tif', height, width, transform, crs)

if __name__ == '__main__':

    T0 = time.time()
    grid_result()

    GVVI()
    T1 = time.time()
    print('Total time:', T1 - T0)
    np.savetxt('data/table/table_1/03_GVVI.csv', [(T1 - T0) / 3600], fmt='%.2f')
