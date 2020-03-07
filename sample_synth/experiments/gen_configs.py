import json
import copy
import os

fp = open("../hyparam.json")
org = json.load(fp)
settings = [
    {"state_type": "discrete", "sampling_type": "gumbel-softmax"},
    {"state_type": "discrete", "sampling_type": "gumbel-max"},
    {"state_type": "discrete", "sampling_type": "none"},
    {"state_type": "normal", "sampling_type": "normal"},
    {"state_type": "normal", "sampling_type": "none"},
    {"dynamics_type": "function"},
    {
        "potential_enabled": True,
        "potential_grad_transition_enabled": False,
        "potential_nn_enabled": False,
    },
    {
        "potential_enabled": False,
        "potential_grad_transition_enabled": False,
        "potential_nn_enabled": False,
    },
    {
        "dynamics_type": "function",
        "potential_enabled": True,
        "potential_grad_transition_enabled": True,
        "potential_nn_enabled": False,
    },
    {
        "dynamics_type": "function",
        "potential_enabled": True,
        "potential_grad_transition_enabled": True,
        "potential_nn_enabled": True,
    },
]
o = {
    "evaluation_output": "sample_synth/experiments/model.%d/hyparam.result.json",
    "load_model": "sample_synth/experiments/model.%d/model.last.ckpt",
    "save_model": "sample_synth/experiments/model.%d/model.last.ckpt",
    "plot_path": "sample_synth/experiments/model.%d/plot",
    "save_model_path": "sample_synth/experiments/model.%d/model",
    "save_result_filter": "sample_synth/experiments/model.%d/result/filter.jbl",
    "save_result_test": "sample_synth/experiments/model.%d/result/test.jbl",
    "save_result_train": "sample_synth/experiments/model.%d/result/train.jbl",
    "simulation_path": "sample_synth/experiments/model.%d/sim",
}

os.makedirs("./configs", exist_ok=True)
for i, el in enumerate(settings):
    os.makedirs("./model.%d/plot" % (i,), exist_ok=True)
    os.makedirs("./model.%d/model" % (i,), exist_ok=True)
    os.makedirs("./model.%d/result" % (i,), exist_ok=True)
    os.makedirs("./model.%d/sim" % (i,), exist_ok=True)
    cfg = copy.deepcopy(org)
    cfg.update(el)
    for k, v in o.items():
        cfg[k] = v % (i,)
    fp = open("./configs/config." + str(i) + ".json", "w")
    json.dump(cfg, fp)
