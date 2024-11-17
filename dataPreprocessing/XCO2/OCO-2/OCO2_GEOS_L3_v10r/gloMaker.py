import numpy as np
china = np.load("china_area.npy") > 0.0
globe = np.load("globe.npy")
globe[china] = True
np.save("globalMask.npy", globe)