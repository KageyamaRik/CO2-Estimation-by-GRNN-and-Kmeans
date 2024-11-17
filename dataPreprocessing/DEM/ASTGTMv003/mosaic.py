# coding=utf-8
# -*- coding:gbk-*-
import arcpy
import os
from arcpy import env

arcpy.CheckOutExtension("Spatial")  # 设置工作空间，即上一步tif影像的保存目录
env.workspace = r'H:\ASTGTMv003'  # 输入路径
tif_path = r'H:\ASTGTMv003'
out_path = r'H:\ASTGTMv003\mosaic'

data_files = [i for i in os.listdir(tif_path) if i.endswith("dem.tif")]
coordinate = arcpy.SpatialReference(4326)
filename = "ASTGTMv003_mosaic.tif"
arcpy.MosaicToNewRaster_management(data_files, out_path, filename, coordinate, "16_BIT_SIGNED", None, "1",
                                   "MEAN", "FIRST")
