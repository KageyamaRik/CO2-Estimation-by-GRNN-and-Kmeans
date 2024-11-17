import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.rcParams['axes.unicode_minus'] = False    # 负号

width = 360
height = 180
inPath = 'H:\\SIF\\GOSIF_v2\\Annual\\Aggregate'
outPath = 'H:\\SIF\\GOSIF_v2\\Annual\\npy'
# 南北极填充，不需要则在循环中注释掉
NFill = np.ones((15, 360)) * -9999.0
SFill = np.ones((25, 360)) * -9999.0
# 绘制数据范围：
vmin = -10.0
vmax = 50.0

if not os.path.exists(outPath):
    os.mkdir(outPath)

for file in os.listdir(inPath):
    if file.endswith("tif"):
        tif = gdal.Open(inPath + "\\" + file)
        band = tif.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, width, height)
        data[data < -1000.0] = -9999.0
        # data = np.vstack((NFill, data, SFill))
        data = np.flip(data, axis=0)  # 磁场颠佬
        np.save(outPath + "\\" + file.replace("tif", "npy"), data)

# 绘制测试：
test = np.load(outPath + "\\" + os.listdir(outPath)[0])

longitude = np.linspace(-179.5, 179.5, 360)
latitude = np.linspace(-89.5, 89.5, 180)
lon, lat = np.meshgrid(longitude, latitude)

plt.rcParams['figure.figsize'] = (12.0, 8.0)
Map = Basemap(llcrnrlon=-180., llcrnrlat=-90., urcrnrlon=180., urcrnrlat=90.)
Map.drawcoastlines()
parallels = np.arange(-75., 76, 25.)
# labels = [left,right,top,bottom]
Map.drawparallels(parallels, labels=[True, False, False, False])
meridians = np.arange(0., 361., 60.)
Map.drawmeridians(meridians, labels=[False, False, False, True])

x, y = Map(lon, lat)

mapPlot = Map.pcolor(x, y, test, cmap='viridis', vmin=vmin, vmax=vmax)
mapPlot.cmap.set_under('w')  # 设置底色
cbar = Map.colorbar(mapPlot, fraction=0.023)
plt.show()
