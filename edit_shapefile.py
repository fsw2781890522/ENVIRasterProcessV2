import os
from osgeo import ogr
from utils import *
from tqdm import tqdm
import spectral.io.envi as e
import pandas as pd
from pprint import pprint

shp_dir = r'H:\20240214_ASTER_Group1\ASTER_range_shp'
img_dir = r'H:\20240214_ASTER_Group1\ASTER_range'
# csv_fn = r"H:\20240214_ASTER_Group1\ASTER_MapSheets_uniq_num.csv"

# df = pd.read_csv(csv_fn)
# # 转换数据框为字典
# data_dict = df.to_dict(orient='list')
# pprint(data_dict)

# # 如果您想要使用第一列的值作为键，第二列的值作为对应键的值，您可以创建一个新的字典
# new_dict = {}
# for key, value in zip(data_dict['Image Name'], data_dict['Map Names']):
#     new_dict[key] = value
# pprint(new_dict)

shp_fn_list = get_fn_list(shp_dir, ext='shp')

for shp_fn in tqdm(shp_fn_list):

    img_fn = shp_fn.replace(shp_dir, img_dir).replace('.shp', '.hdr')
    raster = e.open(img_fn)
    meta = raster.metadata

    # 打开Shapefile数据源
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_fn, 1)  # 第二个参数1表示以写入模式打开

    # 获取图层
    layer = dataSource.GetLayer()

    # 添加字段
    fields_to_add = ['Filename', 'AcqDate', 'MapSheets', 'QuiProblem', 'Remarks']
    for field in fields_to_add:
        if layer.FindFieldIndex(field, 1) == -1:
            layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))

    # 获取去除后缀名的文件名
    original_file_name = os.path.splitext(os.path.basename(shp_fn))[0]

    # 获取 "OriginalFileName" 字段的索引
    Filename_index = layer.GetLayerDefn().GetFieldIndex("Filename")
    AcqDate_index = layer.GetLayerDefn().GetFieldIndex("AcqDate")
    MapSheets_index = layer.GetLayerDefn().GetFieldIndex("MapSheets")
    QuiProblem_index = layer.GetLayerDefn().GetFieldIndex("QuiProblem")
    Remarks_index = layer.GetLayerDefn().GetFieldIndex("Remarks")

    # 开启编辑模式
    layer.StartTransaction()

    # 更新字段值
    for feature in layer:
        feature.SetField(Filename_index, original_file_name.replace('_pro', ''))
        time = (meta['acquisition time'].split('T'))[0]
        feature.SetField(AcqDate_index, time)
        # feature.SetField(CloudCover_index, str(meta['cloud cover']))
        # feature.SetField(MapSheets_index, new_dict[original_file_name])
        layer.SetFeature(feature)

    # 提交编辑
    layer.CommitTransaction()

    # 保存并关闭数据源
    dataSource = None
    raster = None
