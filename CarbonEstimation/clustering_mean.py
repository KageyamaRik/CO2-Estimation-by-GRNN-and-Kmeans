from datasetsLoad import data_load
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
china_shp_path = ("H:\\Region\\区划\\省", "省")


def china_clustering(cl_names, k, coordinate_weight=0.0, cmap='tab10', output=True, n_init=100):

    data = data_load("LAT_year")
    data_cl = {}

    for key in cl_names:
        data_cl[f'{key}'] = data[key][:-1].mean(axis=0)  # 此处修改聚类使用的年份：只用前六年

    longitude = np.linspace(0, 359, 360)
    latitude = np.linspace(0, 179, 180)
    lon, lat = np.meshgrid(longitude, latitude)
    data_cl["lon"] = lon
    data_cl["lat"] = lat

    china = np.load("seven_parts.npy")

    china_bool = china >= 0
    x = np.vstack([data_cl[key][china_bool] for key in data_cl]).T
    length = x.shape[1] - 2
    x[:, 0:length] = (x[:, 0:length] - x[:, 0:length].mean(axis=0)) / x[:, 0:length].std(axis=0)

    clustering = KMeans(n_clusters=k, n_init=n_init)
    weight = np.ones(x.shape[1])
    coordinate_weight = coordinate_weight
    weight[-1] = coordinate_weight
    weight[-2] = coordinate_weight
    clustering.fit(x * weight)
    labels = clustering.labels_
    colors = np.zeros((180, 360), dtype=int) - 1
    for i in range(x.shape[0]):
        colors[int(x[i, -1]), int(x[i, -2])] = labels[i]

    if output:
        longitude = np.linspace(-179.5, 179.5, 360)
        latitude = np.linspace(-89.5, 89.5, 180)
        lon, lat = np.meshgrid(longitude, latitude)

        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        Map = Basemap(llcrnrlon=70., llcrnrlat=16., urcrnrlon=140., urcrnrlat=55.)
        parallels = np.arange(-75., 76., 25.)
        # labels = [left,right,top,bottom]
        Map.drawparallels(parallels, labels=[True, False, False, False])
        meridians = np.arange(0., 361., 60.)
        Map.drawmeridians(meridians, labels=[False, False, False, True])
        Map.readshapefile(*china_shp_path)

        map_x, map_y = Map(lon, lat)

        mapPlot = Map.pcolor(map_x, map_y, colors, cmap=cmap, vmin=0.0, vmax=float(k+1))
        mapPlot.cmap.set_under('w')  # 设置底色
        plt.show()

    return colors, clustering.cluster_centers_
