# MappingXCO2; MOD17A3HGFv061; GOSIF_v2; LandScan_global; VIIRS_annual_v21; ODIAC.
import numpy as np
from math import radians, sin
from maplot import *

options = {"Mapping": "MappingXCO2",
           "GEOS": "OCO2_GEOS_L3_v10r",
           "Lite": "OCO2_L2_Lite_FP_11.1r"}

XCO2 = options["GEOS"]

dataPath = {"MappingXCO2": f"H:\\XCO2\\{XCO2}\\",
            "MOD17A3HGFv061": "H:\\NPP\\MOD17A3HGFv061\\npy\\",
            "LandScan_global": "H:\\Population\\LandScan_global\\npy_sum\\",
            # "VIIRS_annual_v21": "H:\\Nighttime Light\\VIIRS_annual_v21\\npy\\",
            "ODIAC": "H:\\CO2\\ODIAC\\2022\\year\\"}

# [(gC/m2/d * 10^6)tC/m2/d / (area km^2 * 10^6)m^2] * 365days = t (year)
transRaster = np.load("china_area.npy")
np.seterr(divide='ignore', invalid='ignore')


def data_load(filename):
    data = {}
    for key in dataPath:
        if dataPath.get("MappingXCO2"):
            dataPath["MappingXCO2"] = f"H:\\XCO2\\{XCO2}\\" + filename + "\\"
        files = [dataPath[key] + f"{key}_{y}.npy" for y in range(2015, 2022)]
        data_key = np.array([np.load(f) for f in files])
        if key == "LandScan_global":
            data_key = data_key.astype(float)
            data_key /= transRaster
        data[key] = data_key

    return data
