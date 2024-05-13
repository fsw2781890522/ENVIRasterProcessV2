# coding=utf-8
import arcpy
import os
import numpy as np
import spectral.io.envi as e
# from osgeo import ogr
from utils import *
from tqdm import tqdm
from imageprocess import ENVIRaster

# 设置工作环境
# arcpy.env.workspace = "your_workspace_path"

raster_dir = r'H:\20240214_ASTER_Group1\PyPreprocessed'
binary_dir = r'H:\20240214_ASTER_Group1\ASTER_range'
shp_dir = r'H:\20240214_ASTER_Group1\ASTER_range_shp'

if not os.path.exists(binary_dir):
    os.makedirs(binary_dir)
if not os.path.exists(shp_dir):
    os.makedirs(shp_dir)

# csv_fn = r"H:\20240214_ASTER_Group1\ASTER_group1_HDF_Index\ASTER_group1_HDF_Index.csv"
raster_list = get_fn_list(raster_dir, ext='hdr')
for fn in tqdm(raster_list):
    # fn = fn.replace('.hdr', '_pro.hdr')
    basename = os.path.basename(fn)
    o_fn = os.path.join(binary_dir, basename)
    if os.path.exists(o_fn):
        continue
    band1 = ENVIRaster(fn).read_band(0)
    background = (band1 == 0)
    nl, ns, nb = ENVIRaster(fn).raster.shape

    meta = ENVIRaster(fn).raster.metadata.copy()
    new_meta = dict()
    for key in meta.keys():
        value = meta[key]
        if isinstance(value, list) and len(value) == nb:
            new_meta[key] = value[0]
        else:
            new_meta[key] = value
    new_meta['data ignore value'] = 0

    spec_keys = ['wavelength', 'fwhm', 'wavelength units', 'data gain values', 'data offset values', 'bbl']
    for spec_key in spec_keys:
        if spec_key in meta:
            new_meta.pop(spec_key)

    binary_data = np.ones((nl, ns), dtype=np.int16)
    binary_data[background] = 0

    e.save_image(o_fn, binary_data, metadata=new_meta)

binary_list = get_fn_list(binary_dir, ext='img')

for fn in tqdm(binary_list):
    # 输出矢量范围数据集
    out_feature_class = fn.replace(binary_dir, shp_dir).replace('.img', '.shp')
    if os.path.exists(out_feature_class):
        continue
    # print(out_feature_class)

    # 调用Raster Domain工具
    arcpy.RasterDomain_3d(fn, out_feature_class, "POLYGON")

# # 设置工作环境
# arcpy.env.workspace = out_dir
#
# # 获取工作空间中的所有 Shapefile 文件
# shapefiles = arcpy.ListFeatureClasses("*.shp")
#
# # 定义要添加的字段
# fields_to_add = ['NewImagesID', 'OriginalFileName', 'DateOfAcquisition', 'CloudCover', 'MapSheets']
#
# for shpfile in shapefiles:
#     # 获取去除后缀名的文件名
#     original_file_name = os.path.splitext(os.path.basename(shpfile))[0]
#
#     # 添加字段
#     for field in fields_to_add:
#         arcpy.AddField_management(shpfile, field, "TEXT")
#
#     # 更新字段值
#     with arcpy.da.UpdateCursor(shpfile, ['OriginalFileName']) as cursor:
#         for row in cursor:
#             row[0] = original_file_name
#             cursor.updateRow(row)


# 指定Shapefile文件路径


# import os
# from osgeo import gdal, ogr, osr
#
#
# def raster_extent(raster_file):
#     raster = gdal.Open(raster_file)
#     transform = raster.GetGeoTransform()
#     min_x = transform[0]
#     max_x = transform[0] + raster.RasterXSize * transform[1]
#     min_y = transform[3] + raster.RasterYSize * transform[5]
#     max_y = transform[3]
#
#     return (min_x, max_x, min_y, max_y)
#
#
# def raster_to_polygon(raster_file, shp_file):
#     # Open raster dataset
#     raster = gdal.Open(raster_file)
#
#     # Get spatial reference
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)  # WGS84
#
#     # Create output shapefile
#     driver = ogr.GetDriverByName("ESRI Shapefile")
#     shp_ds = driver.CreateDataSource(shp_file)
#     layer = shp_ds.CreateLayer("extent", srs, ogr.wkbPolygon)
#     layer_defn = layer.GetLayerDefn()
#
#     # Get raster extent
#     min_x, max_x, min_y, max_y = raster_extent(raster_file)
#
#     # Create polygon geometry
#     ring = ogr.Geometry(ogr.wkbLinearRing)
#     ring.AddPoint(min_x, min_y)
#     ring.AddPoint(max_x, min_y)
#     ring.AddPoint(max_x, max_y)
#     ring.AddPoint(min_x, max_y)
#     ring.AddPoint(min_x, min_y)
#     poly = ogr.Geometry(ogr.wkbPolygon)
#     poly.AddGeometry(ring)
#
#     # Create feature and write to shapefile
#     feat = ogr.Feature(layer_defn)
#     feat.SetGeometry(poly)
#     layer.CreateFeature(feat)
#
#     # Cleanup
#     shp_ds = None
#     raster = None
#
#
# def batch_raster_to_polygon(input_folder, output_folder):
#     # Ensure output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Get list of raster files
#     raster_files = [f for f in os.listdir(input_folder) if f.endswith('.img')]
#
#     # Process each raster file
#     for raster_file in raster_files:
#         input_path = os.path.join(input_folder, raster_file)
#         output_path = os.path.join(output_folder, os.path.splitext(raster_file)[0] + '.shp')
#         raster_to_polygon(input_path, output_path)
#
#
# if __name__ == '__main__':
#     # Example usage
#     input_folder = 'path/to/your/input/folder'
#     output_folder = 'path/to/your/output/folder'
#     batch_raster_to_polygon(input_folder, output_folder)
