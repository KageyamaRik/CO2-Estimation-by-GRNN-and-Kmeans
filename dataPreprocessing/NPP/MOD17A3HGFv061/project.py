# coding=utf-8
import os
import arcpy
from arcpy import env
from arcpy.sa import *

sourceDir = 'H:\\NPP\\MOD17A3HGFv061\\mosaic'  # 输入
targetDir = 'H:\\NPP\\MOD17A3HGFv061\\wgs1984'  # 输出
# 检查Spatial —ArcGIS Spatial Analyst 扩展模块是否许可
arcpy.CheckOutExtension("Spatial")
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
    # arcpy.ExtractSubDataset_management(tif, outPath, "1")
    arcpy.ProjectRaster_management(tif, outPath, arcpy.SpatialReference(4326))
