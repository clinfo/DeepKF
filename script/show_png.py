import os
import sys
import numpy as np
from build_npy import saveImg

obj = np.load(sys.argv[1])
print obj.shape

saveImg(obj[:, 64, :], "show.png")
