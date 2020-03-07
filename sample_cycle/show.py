import joblib
import sys

o = joblib.load(sys.argv[1])
print(o.keys())
for k, v in o.items():
    if type(v) == list:
        print(k, v[0].shape)
    else:
        print(k, v.shape)
