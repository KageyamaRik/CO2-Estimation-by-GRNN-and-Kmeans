import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import ShuffleSplit

from datasetsLoad import data_load
from myModel.batchCudaGRNN import GRNN
from maplot import *

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

china_shp_path = ("H:\\Region\\区划\\省", "省")
nine_shp_path = ("H:\\Region\\全国shp\\最新2021年全国行政区划\\九段线", "九段线")


class CarbonInverse:

    def __init__(self, raster: np.ndarray, filename, centers=None, input_option=False):

        self.gross_linear_func_params = None
        self.gross_linear_func = None
        self.gross_mae = None
        self.gross_r2 = None
        self.gross_y_test = None
        self.gross_pred = None
        self.anoCoe = None
        self.linear_func_params = None
        self.linear_func = None
        self.pred_colors = None
        self.pred_labels = None
        raster = raster.astype(int)
        self.mae = None
        self.r2 = None
        self.models = []
        self.pred = []
        self.pred_map = np.zeros((180, 360))
        self.y_test_map = np.zeros((180, 360))
        self.raster = raster
        self.centers = centers
        self.num_parts = np.unique(raster).shape[0] - 1  # Number of partitions
        self.input_option = input_option

        data = data_load(filename)  # Datasets
        if input_option:
            data = {key: data[key] for key in data if (key in input_option or key == "ODIAC")}
        # map_bool = np.full((7, 180, 360), True, dtype=bool)
        # for value in data.values():
        #     map_bool = np.logical_and(map_bool, value != -9999.0)
        data_x = {key: data[key] for key in data if key != "ODIAC"}
        data_y = data["ODIAC"]
        cut_train = np.expand_dims(raster, axis=0).repeat(7, axis=0)
        # cut_train[~map_bool] = -1.0
        # if self.centers is None:
        #     self.centers = np.zeros((self.num_parts, 2))  # npp, odiac
        #     for i in range(self.num_parts):
        #         self.centers[i, 0] = data_x["MOD17A3HGFv061"][cut_train == i].mean()
        #         self.centers[i, 1] = data_y[cut_train == i].mean()

        self.x_train = {}
        self.y_train = {}

        # self.x_vali = {}
        # self.y_vali = {}

        longitude = np.linspace(0, 359, 360)
        latitude = np.linspace(0, 179, 180)
        lon, lat = np.meshgrid(longitude, latitude)
        x_test = np.vstack([data_x[key][-1][cut_train[-1] != -1] for key in data_x]).T
        self.x_test = torch.tensor(x_test, dtype=torch.float64)
        y_test = data_y[-1][cut_train[-1] != -1][:, np.newaxis]
        self.y_test = torch.tensor(y_test, dtype=torch.float64)
        self.lon_test = lon[cut_train[-1] != -1]
        self.lat_test = lat[cut_train[-1] != -1]

        # 前六年ODIAC数据均值用于预测异常check
        y_check = data_y[:-1].mean(axis=0)[cut_train[-1] != -1][:, np.newaxis]
        self.y_check = torch.tensor(y_check, dtype=torch.float64)

        for i in range(self.num_parts):
            # 0:-2变更为0:-1
            xtr = np.vstack([data_x[key][0:-1][cut_train[0:-1] == i] for key in data_x]).T
            ytr = data_y[0:-1][cut_train[0:-1] == i][:, np.newaxis]
            self.x_train[i] = torch.tensor(xtr, dtype=torch.float64)
            self.y_train[i] = torch.tensor(ytr, dtype=torch.float64)

            """
            xva = np.vstack([data_x[key][-2][cut_train[-2] == i] for key in data_x]).T
            yva = data_y[-2][cut_train[-2] == i][:, np.newaxis]
            self.x_vali[i] = torch.tensor(xva, dtype=torch.float64)
            self.y_vali[i] = torch.tensor(yva, dtype=torch.float64)
            """

    def cross_validation(self, sigma_range: tuple, n_splits=10, test_size=0.1):
        # 任务：对每个分区的训练集，单独随机划分为n_splits个子集，每个子集的测试集占比为test_size，通过交叉验证求得每个分区的最佳σ值
        # self.x_train结构：{0: [[...], [...]], 1:...}，self.y_train结构：{0: [[.], [.]], 1:...}
        sigmas = []  # 每个分区的sigma值
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
        for i in range(self.num_parts):
            X = self.x_train[i]
            y = self.y_train[i]
            # 新子集结构：X_train, X_test, y_train, y_test
            folds = [[X[tr], X[te], y[tr], y[te]] for (tr, te) in ss.split(X, y)]
            mls = [GRNN() for _ in range(n_splits)]  # 每个子集对应一个临时GRNN模型

            mse_min = float('inf')
            sig_best = torch.tensor(0.0)
            for s in np.linspace(*sigma_range):
                sig = torch.tensor(s)
                for j in range(n_splits):
                    mls[j].fit(folds[j][0], folds[j][2], sig)  # 对每个子模型进行训练，构建神经网络
                preds = [mls[j].predict(folds[j][1]) for j in range(n_splits)]  # 每个子模型对应的交叉验证预测结果
                preds = [np.nan_to_num(pred) for pred in preds]  # 预测出nan值代表在exp()步骤out非常接近0，此处认为0
                # 需重新思考预测异常的原因和应对方法
                tests = [folds[j][3].reshape(-1) for j in range(n_splits)]  # 与preds对应的真值
                MSE = mean_squared_error(np.concatenate(preds), np.concatenate(tests))
                if MSE < mse_min:
                    mse_min = MSE
                    sig_best = sig

            sigmas.append(sig_best)  # 已经是tensor

        return sigmas

    def models_training(self, line_space: tuple, output=True, n_splits=10, test_size=0.1):

        self.models = []
        sigmas = self.cross_validation(line_space, n_splits, test_size)
        for i in range(self.num_parts):
            model = GRNN()
            sig = sigmas[i]

            """
            mse_min = float('inf')
            sigma_best = 0.0

            # If the model does not have the corresponding training data
            if self.x_train[i].shape[0] == 0:
                model.fit(self.x_train[i], self.y_train[i], torch.tensor(0.25))
                self.models.append(model)
                print(f"The best Hyper-parameter (sigma) of Model {i:>2}: No Data")
                continue

            for sig in np.linspace(*line_space):
                sig_torch = torch.tensor(sig)
                model.fit(self.x_train[i], self.y_train[i], sig_torch)
                pred = model.predict(self.x_vali[i])
                pred = np.nan_to_num(pred)  # 预测出nan值代表在exp()步骤out非常接近0，此处认为0
                MSE = mean_squared_error(pred, self.y_vali[i].reshape(-1))
                if MSE < mse_min:
                    mse_min = MSE
                    sigma_best = sig
            """

            model.fit(self.x_train[i], self.y_train[i], sig)
            self.models.append(model)
            if output:
                print(f"The best Hyper-parameter (sigma) of Model {i:>2}: {float(sig)}")

    def prediction(self, output=True, cmap='tab10', alpha=0.3, vmax=None, title=None,
                   xlim=23.0, ylim=22.0, figsize=(6, 5), save=None, s=20):

        if vmax is None:
            vmax = float(self.num_parts)

        self.pred = []
        self.pred_labels = []
        self.pred_colors = []
        length = self.x_test.shape[0]
        for index in range(length):
            label = self.raster[int(self.lat_test[index]), int(self.lon_test[index])]
            self.pred_labels.append(label)
            t = self.x_test[index].unsqueeze(0)
            p = self.models[label].predict(t)
            self.pred.append(p)

        self.pred = torch.tensor(self.pred)
        self.pred = np.nan_to_num(self.pred)

        self.pred_colors = [self.get_color_from_cmap(cmap, 0.0, vmax, i) for i in self.pred_labels]

        self.r2 = r2_score(self.y_test.reshape(-1), self.pred.reshape(-1))
        self.mae = mean_absolute_error(self.y_test.reshape(-1), self.pred.reshape(-1))
        self.linear_func = lambda x, k, b: k * x + b
        self.linear_func_params = curve_fit(self.linear_func, self.y_test.reshape(-1), self.pred.reshape(-1))[0]

        if output:
            plt.figure(figsize=figsize)
            plt.grid(False)
            plt.plot(np.linspace(-5.0, 25.0, 2), np.linspace(-5.0, 25.0, 2), c='b')
            plt.scatter(self.y_test.reshape(-1), self.pred.reshape(-1),
                        alpha=alpha, c=self.pred_colors, edgecolors=None, s=s)
            plt.plot(np.linspace(-0.5, 21, 2), self.linear_func(np.linspace(-0.5, 21, 2), *self.linear_func_params),
                     c='r', linestyle='--')
            plt.ylabel("Estimate (gC/m$^2$/day)")
            plt.xlabel("ODIAC (gC/m$^2$/day)")
            plt.xlim(-1.0, xlim)
            plt.ylim(-1.0, ylim)
            if self.linear_func_params[1] < 0.0:
                plt.annotate(r"y = {:.3f}x - {:.3f}".format(self.linear_func_params[0], -self.linear_func_params[1]),
                             (0.2, 20.5))
            else:
                plt.annotate(r"y = {:.3f}x + {:.3f}".format(*self.linear_func_params), (0.2, 20.5))
            plt.annotate(r"R$^2$ = {:.3f}".format(self.r2), (0.2, 19.25))
            plt.annotate(r"MAE = {:.3f}".format(self.mae), (0.2, 18.0))
            if title:
                plt.title(title)
            if save:
                plt.savefig(save, dpi=600, bbox_inches='tight')
            plt.show()

        indexMax = (self.y_check.reshape(-1) - self.pred.reshape(-1)).argmax()
        self.anoCoe = (self.y_check.reshape(-1) - self.pred.reshape(-1))[indexMax] / self.y_check.reshape(-1)[indexMax]

    # 预测异常检查/无真值校验
    def prediction_anomaly_check(self, output=True, cmap='tab10', alpha=0.3, vmax=None, title=None, xlim=23.0,
                                 ylim=22.0):

        if vmax is None:
            vmax = float(self.num_parts)

        self.pred = []
        self.pred_labels = []
        self.pred_colors = []
        length = self.x_test.shape[0]
        for index in range(length):
            label = self.raster[int(self.lat_test[index]), int(self.lon_test[index])]
            self.pred_labels.append(label)
            t = self.x_test[index].unsqueeze(0)
            p = self.models[label].predict(t)
            self.pred.append(p)

        self.pred = torch.tensor(self.pred)
        self.pred = np.nan_to_num(self.pred)

        self.pred_colors = [self.get_color_from_cmap(cmap, 0.0, vmax, i) for i in self.pred_labels]

        self.r2 = r2_score(self.y_check.reshape(-1), self.pred.reshape(-1))
        self.mae = mean_absolute_error(self.y_check.reshape(-1), self.pred.reshape(-1))
        self.linear_func = lambda x, k, b: k * x + b
        self.linear_func_params = curve_fit(self.linear_func, self.y_check.reshape(-1), self.pred.reshape(-1))[0]

        if output:
            plt.figure(figsize=(6, 5))
            plt.grid(False)
            plt.plot(np.linspace(-5.0, 25.0, 2), np.linspace(-5.0, 25.0, 2), c='b')
            plt.scatter(self.y_check.reshape(-1), self.pred.reshape(-1),
                        alpha=alpha, c=self.pred_colors, edgecolors=None)
            plt.plot(np.linspace(-0.5, 21, 2), self.linear_func(np.linspace(-0.5, 21, 2), *self.linear_func_params),
                     c='r', linestyle='--')
            plt.ylabel("Estimate (gC/m$^2$/day)", weight='bold')
            plt.xlabel("ODIAC: mean of 2014~2019 (gC/m$^2$/day)", weight='bold')
            plt.xlim(-1.0, xlim)
            plt.ylim(-1.0, ylim)
            plt.annotate(r"y = {:.3f}x + {:.3f}".format(*self.linear_func_params), (0.2, 20.5))
            plt.annotate(r"R$^2$ = {:.6f}".format(self.r2), (0.2, 19.25))
            plt.annotate(r"MAE = {:.6f}".format(self.mae), (0.2, 18.0))
            if title:
                plt.title(title)
            plt.show()

        indexMax = (self.y_check.reshape(-1) - self.pred.reshape(-1)).argmax()
        self.anoCoe = (self.y_check.reshape(-1) - self.pred.reshape(-1))[indexMax] / self.y_check.reshape(-1)[indexMax]

    def error_plot(self, cmap='RdBu', vmin=-5.0, vmax=5.0, output=True):

        for i in range(len(self.pred)):
            self.pred_map[int(self.lat_test[i]), int(self.lon_test[i])] = self.pred[i]
        for i in range(self.y_test.shape[0]):
            self.y_test_map[int(self.lat_test[i]), int(self.lon_test[i])] = self.y_test[i]

        diff_map = self.pred_map - self.y_test_map

        if output:
            longitude = np.linspace(-179.5, 179.5, 360)
            latitude = np.linspace(-89.5, 89.5, 180)
            lon, lat = np.meshgrid(longitude, latitude)

            plt.rcParams['figure.figsize'] = (12.0, 8.0)
            Map = Basemap(llcrnrlon=70., llcrnrlat=16., urcrnrlon=140., urcrnrlat=55.)
            parallels = np.arange(-75., 76, 25.)

            # labels = [left,right,top,bottom]
            Map.drawparallels(parallels, labels=[True, False, False, False])
            meridians = np.arange(0., 361., 60.)
            Map.drawmeridians(meridians, labels=[False, False, False, True])
            Map.readshapefile(*china_shp_path)
            Map.readshapefile(*nine_shp_path)

            x, y = Map(lon, lat)

            mapPlot = Map.pcolor(x, y, diff_map, cmap=cmap, vmin=vmin, vmax=vmax)
            mapPlot.cmap.set_under('w')  # 设置底色
            cbar = Map.colorbar(mapPlot, fraction=0.023)
            plt.title("Error: Prediction - Actual Value")
            plt.show()

    def gross_prediction(self, output=True, alpha=0.3, title=None, xlim=None, ylim=None):
        transRaster = np.load("china_area.npy")
        self.gross_pred = (self.pred_map * transRaster)[self.raster >= 0]
        self.gross_y_test = (self.y_test_map * transRaster)[self.raster >= 0]

        self.gross_r2 = r2_score(self.gross_y_test, self.gross_pred)
        self.gross_mae = mean_absolute_error(self.gross_y_test, self.gross_pred)
        self.gross_linear_func = lambda x, k, b: k * x + b
        self.gross_linear_func_params = curve_fit(self.linear_func, self.gross_y_test, self.gross_pred)[0]

        if output:
            plt.figure(figsize=(6, 5))
            plt.grid(False)
            plt.plot(np.linspace(-5.0, 8e7, 2), np.linspace(-5.0, 8e7, 2), c='b')
            plt.scatter(self.gross_y_test, self.gross_pred,
                        alpha=alpha, c=self.pred_colors, edgecolors=None)
            plt.plot(np.linspace(-0.5, 8e7, 2), self.linear_func(np.linspace(-0.5, 8e7, 2), *self.linear_func_params),
                     c='r', linestyle='--')
            plt.ylabel("Estimate (t)", weight='bold')
            plt.xlabel("ODIAC (t)", weight='bold')
            plt.xlim(-1.0, xlim)
            plt.ylim(-1.0, ylim)
            plt.annotate(r"y = {:.3f}x + {:.3f}".format(*self.linear_func_params), (0.2e7, 7.25e7))
            plt.annotate(r"R$^2$ = {:.6f}".format(self.gross_r2), (0.2e7, 6.75e7))
            plt.annotate(r"MAE = {:.6f}".format(self.gross_mae), (0.2e7, 6.25e7))
            if title:
                plt.title(title)
            plt.show()

    def parts_plot(self, cmap='tab10', vmax=None):
        if vmax is None:
            vmax = float(self.num_parts)
        longitude = np.linspace(-179.5, 179.5, 360)
        latitude = np.linspace(-89.5, 89.5, 180)
        lon, lat = np.meshgrid(longitude, latitude)

        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        Map = Basemap(llcrnrlon=70., llcrnrlat=16., urcrnrlon=140., urcrnrlat=55.)
        parallels = np.arange(-75., 76, 25.)

        # labels = [left,right,top,bottom]
        Map.drawparallels(parallels, labels=[True, False, False, False])
        meridians = np.arange(0., 361., 60.)
        Map.drawmeridians(meridians, labels=[False, False, False, True])
        Map.readshapefile(*china_shp_path)
        Map.readshapefile(*nine_shp_path)

        x, y = Map(lon, lat)

        mapPlot = Map.pcolor(x, y, self.raster, cmap=cmap, vmin=0.0, vmax=vmax)
        mapPlot.cmap.set_under('w')  # 设置底色
        plt.show()

    def parts_plot_surface(self, cmap='tab10', vmax=None):
        if vmax is None:
            vmax = float(self.num_parts)
        longitude = np.linspace(-179.5, 179.5, 360)
        latitude = np.linspace(-89.5, 89.5, 180)
        lon, lat = np.meshgrid(longitude, latitude)

        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        Map = Basemap(projection='aea', resolution='l', lat_0=25.0, lon_0=105.,
                      # llcrnrx=-3500000, llcrnry=-1950000, urcrnrx=3500000, urcrnry=1950000,
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

        mapPlot = Map.pcolor(x, y, self.raster, cmap=cmap, vmin=0.0, vmax=vmax)
        mapPlot.cmap.set_under('w')  # 设置底色
        plt.show()

    def parts_print(self):
        for i in range(self.num_parts):
            print(f"Number of clusters {i:>2}: {(self.raster == i).sum()}")
            if self.centers is not None:
                print(f"Center of clusters {i:>2}: {self.centers[i]}\n")

    @staticmethod
    def get_color_from_cmap(cmap_name, vmin, vmax, value):
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=vmin, vmax=vmax)
        color = cmap(norm(value))
        rgba_color = to_rgba(color)
        return rgba_color
