#
import os
import re
import sys
import tarfile

import numpy as np
import joblib
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import dmm.dmm_input as dmm_input
from dmm.dmm import get_default_config
from dmm.dmm import build_config
import pickle

def run(config,args):
    _, data = dmm_input.load_data(
        config, with_shuffle=True, with_train_test=False
    )
    
    print(data.keys())
    org_shape=data.x.shape
    if config["time_major"]:
        d=org_shape[2]
        x=np.reshape(data.x,(-1,d))
    else:
        print("not imlemented yet: non time_major")
        d=org_shape[2]
        x=np.reshape(data.x,(-1,d))
    dim=config["dim"]
    
    if args.mode=="tsne":
        model = TSNE(n_components=dim, random_state=42)
        z = model.fit_transform(x)
    elif args.mode=="pca":
        model = PCA(n_components=dim)
        if args.data_sampling:
            idx=np.random.randint(len(x),1000)
            model.fit(x[idx,:])
        else:
            model.fit(x)
        z = model.transform(x)
    elif args.mode=="umap":
        model = umap.UMAP(n_neighbors=5, n_components=dim, random_state=42)
        if args.data_sampling:
            idx=np.random.randint(len(x),1000)
            model.fit(x[idx])
        else:
            model.fit(x)
        z = model.transform(x)
    else:
        print("not implemented:",args.mode)
        exit()
    z=np.reshape(z,(org_shape[0],org_shape[1],dim))
    
    results={}
    filename=args.mode+".jbl"
    #filename=config["save_result_pca"]
    #results["x"]=data.x
    results["z_q"]=z
    print("[SAVE]",filename)
    joblib.dump(results, filename, compress=3)
    ###
    save_path=config["save_model_path"]+"/"+args.mode+".pkl"
    print("[SAVE]",save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="pca/tsne/umap")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--model", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--data_sampling", action="store_true", help="sampling data"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="pc2",
    )
    parser.add_argument("--profile", action="store_true", help="")
    args = parser.parse_args()
    # config
    config = get_default_config()
    if args.config is None:
        if not args.no_config:
            parser.print_help()
    else:
        fp = open(args.config, "r")
        config.update(json.load(fp))
    if args.dim:
        config["dim"]=args.dim
    build_config(config)

    run(config,args)
   
if __name__ == "__main__":
    main()
