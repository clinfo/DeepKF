import dmm.hyopt as hy
import os
import sys
import itertools
import random

process_num = 2


if len(sys.argv) > 1 and sys.argv[1] == "rm":
    cnt = 1
    idx = "%05d" % (cnt)
    model_path = "hyopt/model" + idx + ""
    result_path = "hyopt/result" + idx + ""
    try:
        os.removedirs(model_path)
    except OSError:
        pass
    try:
        os.removedirs(result_path)
    except OSError:
        pass
    os.remove("hyopt/hyparam" + idx + ".result.json")
    os.remove("hyopt/hyparam" + idx + ".json")
    quit()


###
param_set = {}
param_set["dim"] = [2, 4, 8]
param_set["emission_internal_layers"] = [
    [{"name": "fc_bn"}, {"name": "do"}, {"name": "fc_bn"}],
    [
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_bn"},
    ],
    [
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_res_start"},
        {"name": "fc_bn"},
        {"name": "fc_res"},
        {"name": "do"},
        {"name": "fc_bn"},
    ],
]
param_set["transition_internal_layers"] = [
    [{"name": "fc_bn"}, {"name": "do"}, {"name": "fc_bn"}],
    [
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_bn"},
    ],
    [
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "fc_res_start"},
        {"name": "fc_bn"},
        {"name": "fc_res"},
        {"name": "do"},
        {"name": "fc_bn"},
    ],
]
param_set["variational_internal_layers"] = [
    [{"name": "fc_bn"}, {"name": "do"}, {"name": "fc_bn"}],
    [{"name": "fc_bn"}, {"name": "do"}, {"name": "lstm"}],
    [
        {"name": "fc_bn"},
        {"name": "do"},
        {"name": "lstm"},
        {"name": "do"},
        {"name": "lstm"},
    ],
]


keys = param_set.keys()
xs = [param_set[k] for k in keys]
# x1=param_set["emssion_internal_layers"]
# x2=param_set["transition_internal_layers"]
# x3=param_set["variational_internal_layers"]

cnt = 0
scripts = []
for l in itertools.product(*xs):
    cnt += 1
    idx = "%05d" % (cnt)
    model_path = "hyopt/model" + idx + ""
    result_path = "hyopt/result" + idx + ""

    hy.initialize_hyperparameter(load_filename="hyopt_hyparam_template.json")
    param = hy.get_hyperparameter()
    param["evaluation_output"] = "hyopt/hyparam" + idx + ".result.json"
    param["hyperparameter_input"] = "hyopt/hyparam" + idx + ".json"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    param["save_model_path"] = model_path
    param["load_model"] = ""
    param["save_result_train"] = result_path + "/train.jbl"
    param["save_result_test"] = result_path + "/test.jbl"
    param["save_result_filter"] = result_path + "/filter.jbl"
    param["plot_path"] = result_path + "/plot"
    param["simulation_path"] = result_path + "/sim"

    # param["emssion_internal_layers"]    =l1
    # param["transition_internal_layers"] =l2
    # param["variational_internal_layers"]=l3

    for el, k in zip(l, keys):
        param[k] = el

    ###
    hy.save_hyperparameter(param["hyperparameter_input"])
    cmd = (
        "python dkf.py --config hyopt/config_train.json --hyperparam "
        + param["hyperparameter_input"]
        + " train > "
        + result_path
        + "log.txt 2>&1"
        + "\n"
    )

    scripts.append(cmd)


random.shuffle(scripts)

fps = []
for pid in range(process_num):
    fp = open("hyopt/run" + str(pid) + ".sh", "w")
    fp.write("export CUDA_VISIBLE_DEVICES=" + str(pid) + "\n")
    fps.append(fp)

for i, line in enumerate(scripts):
    j = i % process_num
    fps[j].write(line)
