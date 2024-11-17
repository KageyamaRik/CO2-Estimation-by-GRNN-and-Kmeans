# coding=utf-8
# -*- coding:gbk-*-
import arcpy
import os
from arcpy import env

sourceDir = 'H:\\NPP\\MOD17A3HGFv061'  # 输入
targetDir = 'H:\\NPP\\MOD17A3HGFv061\\tif'  # 输出
# 检查Spatial —ArcGIS Spatial Analyst 扩展模块是否许可
arcpy.CheckOutExtension("Spatial")
# 设置工作空间
env.workspace = sourceDir
arcpy.env.scratchWorkspace = sourceDir
# 读取栅格列表
hdfList = arcpy.ListRasters('*', 'hdf')

for hdf in hdfList:
    # 替换文件名后缀
    fileName = os.path.basename(hdf).replace('hdf', 'tif')
    print(fileName)
    outPath = targetDir + '\\' + fileName
    arcpy.ExtractSubDataset_management(hdf, outPath, "1")
    # 0: Gross Primary Productivity
    # 1: Net Primary Productivity
    # 2: Quality Control Bits
