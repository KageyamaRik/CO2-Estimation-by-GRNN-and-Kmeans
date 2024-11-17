import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

china_shp_path = ("H:\\Region\\区划\\省", "省")
nine_shp_path = ("H:\\Region\\全国shp\\最新2021年全国行政区划\\九段线", "九段线")


def map_plot(m, cmap, vmin, vmax, title, bar=True, figsize=(8.0, 8.0), lon_west=70.,
             lat_south=16., lon_east=140., lat_north=55., parallels=np.arange(-75., 76, 25.),
             meridians=np.arange(0., 361., 60.), fraction=0.023):
    longitude = np.linspace(-179.5, 179.5, 360)
    latitude = np.linspace(-89.5, 89.5, 180)
    lon, lat = np.meshgrid(longitude, latitude)

    plt.rcParams['figure.figsize'] = figsize
    Map = Basemap(llcrnrlon=lon_west, llcrnrlat=lat_south, urcrnrlon=lon_east, urcrnrlat=lat_north)
    parallels = parallels

    # labels = [left,right,top,bottom]
    Map.drawparallels(parallels, labels=[True, False, False, False])
    meridians = meridians
    Map.drawmeridians(meridians, labels=[False, False, False, True])
    Map.readshapefile(*china_shp_path)
    Map.readshapefile(*nine_shp_path)

    x, y = Map(lon, lat)

    mapPlot = Map.pcolor(x, y, m, cmap=cmap, vmin=vmin, vmax=vmax)
    mapPlot.cmap.set_under('w')  # 设置底色
    if bar:
        cbar = Map.colorbar(mapPlot, fraction=fraction)
    plt.title(title)
    plt.show()


def map_plot_surface(m, cmap, vmin, vmax, title, bar=True, figsize=(12.0, 8.0), lon_west=70.,
                     lat_south=16., lon_east=140., lat_north=55., parallels=np.arange(-75., 76, 25.),
                     meridians=np.arange(0., 361., 60.), fraction=0.023, norm=None, save=None):
    longitude = np.linspace(-179.5, 179.5, 360)
    latitude = np.linspace(-89.5, 89.5, 180)
    lon, lat = np.meshgrid(longitude, latitude)

    plt.rcParams['figure.figsize'] = figsize
    Map = Basemap(projection='aea', resolution='l', lat_0=25.0, lon_0=105.,
                  llcrnrlon=78., llcrnrlat=15., urcrnrlon=138., urcrnrlat=52.
                  )
    parallels = np.arange(-70., 71, 10.)

    # labels = [left,right,top,bottom]
    Map.drawparallels(parallels, labels=[True, False, False, False])
    meridians = np.arange(0., 361., 10.)
    Map.drawmeridians(meridians, labels=[False, False, False, True])
    Map.readshapefile(*china_shp_path, linewidth=0.2, antialiased=True)
    Map.readshapefile(*nine_shp_path)

    x, y = Map(lon, lat)

    mapPlot = Map.pcolor(x, y, m, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    if bar:
        cbar = Map.colorbar(mapPlot, fraction=fraction)
    plt.title(title)
    if save:
        plt.savefig(save, dpi=600, bbox_inches='tight')
    plt.show()


def map_plot_surface_sub(m, cmap, vmin, vmax, title, bar=True, lon_west=70.,
                         lat_south=16., lon_east=140., lat_north=55., parallels=np.arange(-75., 76, 25.),
                         meridians=np.arange(0., 361., 60.), fraction=0.023, norm=None):
    longitude = np.linspace(-179.5, 179.5, 360)
    latitude = np.linspace(-89.5, 89.5, 180)
    lon, lat = np.meshgrid(longitude, latitude)

    Map = Basemap(projection='aea', resolution='l', lat_0=25.0, lon_0=105.,
                  llcrnrlon=78., llcrnrlat=15., urcrnrlon=138., urcrnrlat=52.
                  )
    parallels = np.arange(-70., 71, 10.)

    # labels = [left,right,top,bottom]
    Map.drawparallels(parallels, labels=[True, False, False, False])
    meridians = np.arange(0., 361., 10.)
    Map.drawmeridians(meridians, labels=[False, False, False, True])
    Map.readshapefile(*china_shp_path, linewidth=0.2, antialiased=True)
    Map.readshapefile(*nine_shp_path)

    x, y = Map(lon, lat)

    mapPlot = Map.pcolor(x, y, m, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    if bar:
        cbar = Map.colorbar(mapPlot, fraction=fraction)
    plt.title(title)
