import math
import __hyopt as hy

def get_default_config():
    """
    デフォルトの設定を<dict>configに読み込む関数
    """
    
    config = {}
    
    # data and network
    # config["dim"]=None
    config["dim"] = 2
    
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100 # dmm_mode, train
    config["alpha"] = 1.0
    config["beta"] = 1.0
    config["gamma"] = 1.0
    config["learning_rate"] = 1.0e-2
    config["curriculum_alpha"] = False
    config["curriculum_beta"] = False
    config["curriculum_gamma"] = False
    config["epoch_interval_save"] = 10  # 100
    config["epoch_interval_print"] = 10  # 100
    config["epoch_interval_eval"] = 1
    config["sampling_tau"] = 10  # 0.1
    config["normal_max_var"] = 5.0  # 1.0
    config["normal_min_var"] = 1.0e-5
    config["zero_dynamics_var"] = 1.0
    config["pfilter_sample_size"] = 10
    config["pfilter_proposal_sample_size"] = 1000
    config["pfilter_save_sample_num"] = 100
    
    # dataset
    config["train_test_ratio"] = [0.8, 0.2] # dmm_input
    config["data_train_npy"] = None # dmm_input
    config["mask_train_npy"] = None # dmm_input
    config["label_train_npy"] = None # dmm_input
    config["data_test_npy"] = None # dmm_input
    config["mask_test_npy"] = None # dmm_input
    config["label_test_npy"] = None
    config["label_type"] = "multinominal"
    config["task"] = "generative"
    
    # save/load model
    config["save_model_path"] = None
    config["load_model"] = None
    config["save_result_train"] = None
    config["save_result_test"] = None
    config["save_result_filter"] = None
    
    # config["state_type"]="discrete"
    config["state_type"] = "normal"
    config["sampling_type"] = "none"
    config["time_major"] = True # dmm_input
    config["steps_train_npy"] = None # dmm_input
    config["steps_test_npy"] = None # dmm_input
    config["sampling_type"] = "normal"
    config["emission_type"] = "normal"
    config["state_type"] = "normal"
    config["dynamics_type"] = "distribution" # inference()
    config["pfilter_type"] = "trained_dynamics"
    config["potential_enabled"] = (True,)
    config["potential_grad_transition_enabled"] = (True,)
    config["potential_nn_enabled"] = (False,)
    config["potential_grad_delta"] = 0.1
    
    #
    config["field_grid_num"] = 30
    config["field_grid_dim"] = None
    
    # generate json
    # fp = open("config.json", "w")
    # json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    return config

def get_batch_size(config, data):
    """
    バッチサイズとバッチの数を返す関数
    あらかじめ決めたバッチサイズ>データ数の場合、バッチサイズ=データ数
    """
    
    batch_size = min(data.num, config["batch_size"]) # データ数がバッチサイズより小さい場合、バッチサイズ=データ数、デフォルトでは100
#     n_batch = int(data.num / batch_size) # バッチの数
    n_batch = math.ceil(data.num / config["batch_size"]) # バッチの数、最低1

#     if n_batch == 0: # 
#         batch_size = data.num
#         n_batch = 1
#     elif n_batch * batch_size != data.num:
#         n_batch += 1
    
    return batch_size, n_batch

def get_dim(config, hy_param, data):
    """
    次元を操る関数
    何してる？dimとdim_emitの違いは？
    
    返り値
    dim, dim_emit: 次元
    """
    
    dim_emit = None
    
    if data is not None: # データが与えられなかった場合
        dim_emit = data.dim
    elif "dim_emit" in config: # データが与えられ、configに"dim_emit"がある場合、デフォルトではない
        dim_emit = config["dim_emit"]
    elif "dim_emit" in hy_param: # データが与えられ、hy_paramに"dim_emit"がある場合、デフォルトではない
        dim_emit = hy_param["dim_emit"]
    else: # データが与えられ、configにもhy_paramにも"dim_emit"がない場合
        dim_emit = 1
    
    if config["dim"] is None: # configにdimがない場合、デフォルトではある
        dim = dim_emit # dimを設定
        config["dim"] = dim # configに設定
    else: # configにdimがある場合
        dim = config["dim"]
    
    hy_param["dim"] = dim
    hy_param["dim_emit"] = dim_emit
    
    return dim, dim_emit # 次元を返す


def build_nn(dim_input, n_steps, hyparam_name, name, init_params_flag, config):
    """
    Retruns the layer of the neural networks. 
    Parameters
    ----------
        x :

        dim_input :

        hyparm_name :

    Returns
    -------
        layer :

        layer_dim :

    """

#     wd_bias = None
#     wd_w = 0.1
#     layer = x
    layer_dim = dim_input
#     res_layer = None
    hy_param = hy.get_hyperparameter() # ハイパーパラメータの読み込み

#     print("build_nn: hyparam = ", hy_param)

#     print("build_nn: hyparam[variational_internal_layers] = ", hy_param["variational_internal_layers"])
    
#     print("build_nn: hyparam_name = ",hyparam_name)
    
    """
    build_nn: hyparam =  {'alpha': 1.0, 'batch_size': 100, 'epoch': 100, 'dim': 0, 
    'emission_internal_layers': [{'name': 'fc', 'dim_output': 64}, {'name': 'fc'}], 
    'transition_internal_layers': [{'name': 'fc'}, {'name': 'fc'}], 
    'variational_internal_layers': [{'name': 'fc'}, {'name': 'lstm'}], 
    'potential_internal_layers': [{'name': 'do'}, {'name': 'fc'}], 
    'evaluation': {'all_costs': [0.0, 0.0, 0.0], 'cost': 0.0, 'error': 0.0, 
    'validation_all_costs': [0.0, 0.0, 0.0], 'validation_cost': 0.0}, 
    'evaluation_output': 'model/hyparam.result.json', 'hyperparameter_input': 'sample/hyparam.json', 'plot_path': 'model/plot',
    'save_model_path': 'model/model', 'save_result_filter': 'model/result/filter.jbl', 'save_result_test': 'model/result/test.jbl',
    'save_result_train': 'model/result/train.jbl', 'simulation_path': 'model/sim', 'patience': 0, 'potential_enabled': True,
    'potential_grad_transition_enabled': True, 'potential_nn_enabled': True, 'save_model': 'model/model/model.last.ckpt', 
    'load_model': 'model/model/model.last.ckpt', 'train_test_ratio': [0.8, 0.2], 'dim_emit': 4, 'n_steps': 3}
    
    build_nn: hyparam[variational_internal_layers] =  [{'name': 'fc'}, {'name': 'lstm'}]
    build_nn: hyparam_name =  "variational_internal_layers"
    
    """
    
    for i, hy_layer in enumerate(hy_param[hyparam_name]):
        """
        build_nn: [i, hy_layer] = [0, {'name': 'fc'}]
        build_nn: [i, hy_layer] = [1, {'name': 'lstm'}]                
        """
        
#         print("build_nn: i, hy_layer = {}, {}".format(i,hy_layer))
        
        layer_dim_out = layer_dim # 
#         # print(">>>",layer_dim)
        
        if hyparam_name == "emission_internal_layers": # hyparam_name == "emission_internal_layers"のとき
#         if "dim_output" in hy_layer: # hy_layerにdim_outputがあるとき、というか
            layer_dim_out = hy_layer["dim_output"] # hy_layer["dim_output"]を採用
            # print(">>>",layer_dim,"=>",layer_dim_out)

        if hy_layer["name"] == "fc": # 
            with tf.variable_scope(name + "_fc" + str(i)) as scope:
                layer = layers.fc_layer(
                    "fc" + str(i),
                    layer,
                    layer_dim,
                    layer_dim_out,
                    wd_w,
                    wd_bias,
                    activate=tf.nn.relu,
                    init_params_flag=init_params_flag,
                )
                layer_dim = layer_dim_out
#         elif hy_layer["name"] == "fc_res_start":
#             res_layer = layer
#         elif hy_layer["name"] == "do":
#             dropout_rate = control_params["placeholders"]["dropout_rate"]
#             layer = layers.dropout_layer(layer, dropout_rate)
#             layer_dim = layer_dim_out
#         elif hy_layer["name"] == "fc_res":
#             with tf.variable_scope(name + "_fc_res" + str(i)) as scope:
#                 layer = layers.fc_layer(
#                     name + "_fc_res" + str(i),
#                     layer,
#                     layer_dim,
#                     layer_dim_out,
#                     wd_w,
#                     wd_bias,
#                     activate=None,
#                     init_params_flag=init_params_flag,
#                 )
#                 layer = res_layer + layer
#                 layer = tf.sigmoid(layer)
#                 layer_dim = layer_dim_out
#         elif hy_layer["name"] == "fc_bn":
#             is_train = control_params["placeholders"]["is_train"]
#             with tf.variable_scope(name + "_fc_bn" + str(i)) as scope:
#                 layer = layers.fc_layer(
#                     name + "_fcbn" + str(i),
#                     layer,
#                     layer_dim,
#                     layer_dim_out,
#                     wd_w,
#                     wd_bias,
#                     activate=tf.nn.relu,
#                     with_bn=True,
#                     init_params_flag=init_params_flag,
#                     is_train=is_train,
#                 )
#                 layer_dim = layer_dim_out
#         elif hy_layer["name"] == "cnn":
#             with tf.variable_scope(name + "_cnn" + str(i)) as scope:
#                 layer = tf.reshape(layer, [-1, n_steps, layer_dim])
#                 layer = tf.layers.conv1d(
#                     layer,
#                     layer_dim_out,
#                     1,
#                     padding="SAME",
#                     reuse=(not init_params_flag),
#                     name="conv" + str(i),
#                 )
#                 layer = tf.reshape(layer, [-1, layer_dim_out])
#                 layer_dim = layer_dim_out
#         elif hy_layer["name"] == "lstm":
#             with tf.variable_scope(name + "_lstm" + str(i)) as scope:
#                 layer = tf.reshape(layer, [-1, n_steps, layer_dim])
#                 layer = layers.lstm_layer(
#                     layer, n_steps, layer_dim_out, init_params_flag=init_params_flag
#                 )
#                 layer = tf.reshape(layer, [-1, layer_dim_out])
#                 layer_dim = layer_dim_out
#         else:
#             assert "unknown layer:" + hy_layer["name"]

    return layer, layer_dim