import os
import sys
import numpy as np

obj = np.load(sys.argv[1])
print(obj.shape)
print(np.max(obj))
print(np.min(obj))
# print(obj)
# print(np.mean(obj))
