import numpy as np
import joblib
import json
import sys
import argparse
from dmm.plot_input import load_plot_data, get_default_argparser, load_config
from matplotlib import pylab as plt


def main():
    parser = get_default_argparser()
    parser.add_argument(
        "--obs_dim", type=int, default=0, help="a dimension of observation for plotting"
    )
    parser.add_argument('--num_particle', type=int,
        default=10,
        help='the number of particles (latent variable)')
    parser.add_argument('--obs_num_particle', type=int,
        default=10,
        help='the number of particles (observation)')
    parser.add_argument(
        "--anim",
        action="store_true",
        default=False,
        help="Animation",
    )
 
    args = parser.parse_args()
    # config
    """
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pylab as plt
    else:
        from matplotlib import pylab as plt
    """
    ##
    print("=============")
    print("... loading")
    # d=data_info["attr_emit_list"].index("206010")
    # print("206010:",d)
    if args.config is None:
        parser.print_help()
        quit()
    config = load_config(args)
    data,result_data = load_plot_data(args,config=config)
    print("data:",data.keys())
    print("result:",result_data.keys())
    print("=============")
    if "plot_path" in config:
        out_dir=config["plot_path"]
    filename_result=config["simulation_path"]+"/field.jbl"

    print("[LOAD] ",filename_result)
    obj=joblib.load(filename_result)
    print("field:",obj.keys())
    data_z=obj["z"]
    #data_gz=-obj["gz"][0]
    data_gz=obj["gz"]
    print("shape z:",data_z.shape)
    print("shape grad. z",data_gz.shape)
    #
    #fp = open(filename_info, 'r')
    #data_info = json.load(fp)
    #d=data_info["attr_emit_list"].index("206010")

    X=data_z[:,0]
    Y=data_z[:,1]
    U=data_gz[:,0]
    V=data_gz[:,1]
    R=np.sqrt(U**2+V**2)
    #plt.axes([0.025, 0.025, 0.95, 0.95])
    #plt.quiver(X, Y, U, V, R, units='xy',angles='xy', alpha=.5, scale_units='xy')
    plt.quiver(X, Y, U, V, units='xy',angles='xy', alpha=.5, scale_units='xy',linewidth=.5)
    #plt.quiver(X, Y, U*100, V*100, units='xy',angles='xy', scale_units='xy')
    #plt.quiver(X, Y, U, V)
    #plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
    #r=1.0#20
    #plt.xlim(-r, r)
    #plt.xticks(())
    #plt.ylim(-r,r)
    #plt.yticks(())

    out_filename=out_dir+"/vec.png"
    print("[SAVE] :",out_filename)
    plt.savefig(out_filename)

    plt.show()
    plt.clf()

