# Data Audit Report

Generated: 2026-07-25 00:01:31

## File Inventory

- **relations/SVI.csv**: 5.8 MB — SVI relations (street view)
- **relations/WVI.csv**: 26.2 MB — WVI relations (window view)
- **relations/DVI.csv**: 101.1 MB — DVI relations (drone view)
- **viewpoints/SVI_position_road.csv**: 4.3 MB — SVI viewpoint positions
- **viewpoints/window_view_location.csv**: 39.5 MB — WVI viewpoint positions
- **viewpoints/DVI_position_60.csv**: 3.4 MB — DVI viewpoint positions
- **greenery/df_grid_ply.csv**: 2.8 MB — Greenery grid definitions
- **greenery/veg_raster_buffer_100.tif**: 0.1 MB — Vegetation raster mask
- **reference/GVVI_S.tif**: 0.5 MB — Official GVVI_S reference
- **reference/GVVI_W.tif**: 0.5 MB — Official GVVI_W reference
- **reference/GVVI_D.tif**: 0.5 MB — Official GVVI_D reference
- **reference/GVVI.tif**: 0.5 MB — Official GVVI reference

## Relations Data


### SVI.csv

- Rows: 27597
- Columns: ['grid_idx_x', 'grid_idx_y', 'raster_value', 'grid_coords_min_x', 'grid_coords_min_y', 'grid_coords_max_x', 'grid_coords_max_y', 'R', 'G', 'B', 'ori_ply_name', 'count', 'OBJ_R', 'OBJ_G', 'OBJ_B', 'img_id', 'total_pixel', 'distance_img', 'pixel_wvi']
- Non-null img_id: 18958
- Empty img_id: 8639
- Mean img_ids per row (sample): 13.7

### WVI.csv

- Rows: 27597
- Columns: ['grid_idx_x', 'grid_idx_y', 'raster_value', 'grid_coords_min_x', 'grid_coords_min_y', 'grid_coords_max_x', 'grid_coords_max_y', 'R', 'G', 'B', 'ori_ply_name', 'count', 'OBJ_R', 'OBJ_G', 'OBJ_B', 'img_id', 'total_pixel', 'distance_img', 'pixel_wvi']
- Non-null img_id: 26598
- Empty img_id: 999
- Mean img_ids per row (sample): 66.2

### DVI.csv

- Rows: 27597
- Columns: ['grid_idx_x', 'grid_idx_y', 'raster_value', 'grid_coords_min_x', 'grid_coords_min_y', 'grid_coords_max_x', 'grid_coords_max_y', 'R', 'G', 'B', 'ori_ply_name', 'count', 'OBJ_R', 'OBJ_G', 'OBJ_B', 'img_id', 'total_pixel', 'distance_img', 'pixel_wvi']
- Non-null img_id: 22780
- Empty img_id: 4817
- Mean img_ids per row (sample): 405.2

## Viewpoint Data


### SVI

- Rows: 18096
- Columns: ['id', 'RouteID', 'Route_split_id', 'OBJECTID', 'name', 'pair_id', 'X', 'Y', 'Z', 'lat', 'lon', 'Heading']
- Unique IDs: 18096

### WVI

- Rows: 84936
- Columns: ['name', 'ID', 'bldg_ID', 'X', 'Y', 'Z', 'lat', 'lon', 'Heading', 'normal_x', 'normal_y', 'ROOFLEVEL', 'BASELEVEL', 'line_0_x', 'line_0_y', 'line_1_x', 'line_1_y', 'adjust_x', 'adjust_y', 'adjust_z', 'move_x', 'move_y', 'move_z', 'move_lat', 'move_lon', 'adjust']
- Unique IDs: 84936

### DVI

- Rows: 17424
- Columns: ['id', 'name', 'points_id', 'height_id', 'x', 'y', 'ori_z', 'z', 'heading', 'lat', 'lon', 'heading_id']
- Unique IDs: 17424

## Greenery Data

- Rows: 27597
- Unique (grid_idx_x, grid_idx_y): 27597
- grid_idx_x range: [0, 244]
- grid_idx_y range: [0, 246]
- X range: [835029.3, 837499.3]
- Y range: [818467.6, 820917.6]

## ID Connection Check


### SVI
- Unique view IDs in relations: 12095
- Viewpoint table IDs: 18096
- Successfully connected: 12095
- Missing from viewpoint table: 0

### WVI
- Unique view IDs in relations: 43341
- Viewpoint table IDs: 84936
- Successfully connected: 43341
- Missing from viewpoint table: 0

### DVI
- Unique view IDs in relations: 14879
- Viewpoint table IDs: 17424
- Successfully connected: 14879
- Missing from viewpoint table: 0