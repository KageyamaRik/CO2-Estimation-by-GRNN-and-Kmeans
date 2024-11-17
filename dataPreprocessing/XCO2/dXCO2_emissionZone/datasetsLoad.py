# MappingXCO2; MOD17A3HGFv061; GOSIF_v2; LandScan_global; VIIRS_annual_v21; ODIAC.
import numpy as np

dataPath = {"MappingXCO2": "H:\\XCO2\\MappingXCO2\\MappingXCO2_year\\",
            # "MOD17A3HGFv061": "H:\\NPP\\MOD17A3HGFv061\\npy\\",
            "GOSIF": "H:\\SIF\\GOSIF_v2\\Annual\\npy\\",
            "LandScan_global": "H:\\Population\\LandScan_global\\npy\\",
            "VIIRS_annual_v21": "H:\\Nighttime Light\\VIIRS_annual_v21\\npy\\",
            "ODIAC": "H:\\CO2\\ODIAC\\2022\\year\\"}


def data_load():
    data = {}
    for key in dataPath:
        files = [dataPath[key] + f"{key}_{y}.npy" for y in range(2014, 2021)]
        data_key = np.array([np.load(f) for f in files])
        data[key] = data_key

    return data
