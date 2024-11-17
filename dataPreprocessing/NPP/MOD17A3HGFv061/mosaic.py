# coding=utf-8
# -*- coding:gbk-*-
import arcpy
import os
from arcpy import env
import datetime

arcpy.CheckOutExtension("Spatial")  # 设置工作空间，即上一步tif影像的保存目录
env.workspace = r'H:\NPP\MOD17A3HGFv061\tif'  # 输入路径
tif_path = r'H:\NPP\MOD17A3HGFv061\tif'
out_path = r'H:\NPP\MOD17A3HGFv061\mosaic'
date_file = {}

for i in os.listdir(tif_path):
    if i.endswith('.tif'):
        date_str = int(i.split('.')[1][1:])
        year = date_str // 1000
        if year >= 2015 or year <= 2013:
            continue
        days = date_str % 1000
        tif_date = datetime.date(year, 1, 1) + datetime.timedelta(days - 1)
        if tif_date not in date_file:
            date_file[tif_date] = [i]
        else:
            date_file[tif_date].append(i)
all_dates = date_file.keys()
coordinate = arcpy.SpatialReference(4326)
for tif_date in sorted(all_dates):
    # for tif_date in [datetime.date(2017, 1, 1)]:
    tif_files = date_file[tif_date]
    year = tif_date.year
    days = (tif_date - datetime.date(year, 1, 1)).days + 1
    filename = 'MOD17A3HGF.A%d%s.tif' % (year, str(days).zfill(3))
    # print(tif_date, file_name)
    # out_file = os.path.join(out_path, )
    print(tif_date)
    arcpy.MosaicToNewRaster_management(tif_files, out_path, filename, coordinate, "16_BIT_SIGNED", None, "1",
                                       "MEAN", "FIRST")
