# coding=utf-8
# Name: Aggregate_Ex_02.py
# Description: Generates a reduced resolution version of a raster.
# Requirements: Spatial Analyst Extension

# Import system modules
import arcpy
from arcpy import env
from arcpy.sa import *
import os

sourceDir = 'H:\\NPP\\MOD17A3HGFv061\\setNull'  # 输入
targetDir = 'H:\\NPP\\MOD17A3HGFv061\\aggregate'  # 输出
# 检查Spatial —ArcGIS Spatial Analyst 扩展模块是否许可
arcpy.CheckOutExtension("Spatial")
cellFactor = 240
# 设置工作空间
env.workspace = sourceDir
arcpy.env.scratchWorkspace = sourceDir
# 读取栅格列表
tifList = arcpy.ListRasters('*', 'tif')

# for 循环
for tif in tifList:
    if "2014" not in tif:
        continue
    # print tif  #打印文件名用于检查
    # 替换文件名后缀
    newName = os.path.basename(tif).replace('tif', 'tif')
    outPath = targetDir + '\\' + newName
    print(outPath)   # 打印文件名用于检查,python2或3的打印方式不同
    # Execute Aggregate
    outAggregate = Aggregate(tif, cellFactor, "MEAN", "EXPAND", "DATA")
    # Save the output
    outAggregate.save(outPath)
