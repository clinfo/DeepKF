import __hyopt as hy
import __dmm_input as dmm_input
from __dmm_functions import get_batch_size, get_dim, build_nn

# return parameters for q(z): list of tensors: batch_size x n_steps x dim
def computeVariationalDist(n_steps, init_params_flag=True, config=None):
    """
    わからない
    
    Return parameters for q(z|x_{1:t}).
    Parameters
    ----------
    x : 
        bs x t x dim_emit
        or (bs x t) x dim_emit

    Returns
    -------
    params : list
        layer_z : 
            bs x T x dim
        layer_mu : 
            bs x T x dim
        layer_cov : 
            bs x T x dim
    """
    
    hy_param = hy.get_hyperparameter() # ハイパーパラメータの取得
#     dim = hy_param["dim"]
#     dim_emit = hy_param["dim_emit"]
    wd_bias = None # 
    wd_w = 0.1 # 
#     x = tf.reshape(x, [-1, hy_param["dim_emit"]])
#     x = tf.reshape(x, [-1, hy_param["dim_emit"]])
    
    params = []
    
#     with tf.name_scope("variational_dist") as scope_parent:
#         with tf.variable_scope("variational_dist_var") as v_scope_parent:

    # ニューラルネットワークの構築
    layer, dim_out = build_nn(
        dim_input=hy_param["dim_emit"],
        n_steps=n_steps,
        hyparam_name="variational_internal_layers",
        name="vd",
        init_params_flag=init_params_flag,
        config=config,
    )
    
#     sttype = control_params["config"]["state_type"]

#             if sttype == "discrete" or sttype == "discrete_tr":
#                 with tf.variable_scope("vd_fc_logits") as scope:
#                     layer_logit = layers.fc_layer(
#                         "vd_fc_logits",
#                         layer,
#                         dim_out,
#                         dim,
#                         wd_w,
#                         wd_bias,
#                         activate=None,
#                         init_params_flag=init_params_flag,
#                     )
            
#                 layer_logit = tf.reshape(layer_logit, [-1, n_steps, dim])
#                 layer_z = tf.nn.softmax(layer_logit)
#                 params.append(layer_z)
#             elif sttype == "normal":
#                 with tf.variable_scope("vd_fc_mu") as scope:
#                     layer_mu = layers.fc_layer(
#                         "vd_fc_mu",
#                         layer,
#                         dim_out,
#                         dim,
#                         wd_w,
#                         wd_bias,
#                         activate=None,
#                         init_params_flag=init_params_flag,
#                     )
#                     layer_mu = tf.reshape(layer_mu, [-1, n_steps, dim])
#                     params.append(layer_mu)
                
#                 with tf.variable_scope("vd_fc_cov") as scope:
#                     pre_activate = layers.fc_layer(
#                         "vd_fc_cov",
#                         layer,
#                         dim_out,
#                         dim,
#                         wd_w,
#                         wd_bias,
#                         activate=None,
#                         init_params_flag=init_params_flag,
#                     )
#                     layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
#                     max_var = control_params["config"]["normal_max_var"]
#                     min_var = control_params["config"]["normal_min_var"]
#                     layer_cov = tf.clip_by_value(layer_cov, min_var, max_var)
#                     layer_cov = tf.reshape(layer_cov, [-1, n_steps, dim])
#                     params.append(layer_cov)
#             else:
#                 raise Exception("[Error] unknown state type")

    return params

def sampleVariationalDist(n_steps, init_params_flag=True, config=None):
    """
    わからない
    
    Parameters
    ----------
        x :
            the observed value x_{1:t}.
    Returns
    -------
        qz :
            the parameters of the variational distribution q(z|x)
        qs :
            the state z_s which is sampled from the variational distribution q(z|x)
    """
    qz = computeVariationalDist(n_steps, init_params_flag, config)
    qs = sampleState(qz, config)
    
    return qs, qz


def inference(n_steps, config):
    """
    
    
    Returns
    -------
    outputs :
    indference results
    """
    
    if config["dynamics_type"] in ["distribution","function"]: # config["dynamics_type"]が適切である場合、以下の操作を行う
        pass
    else: # config["dynamics_type"]が適切でない場合、エラーを発生して終了
        raise Exception("[Error] unknown dynamics type")
    
    # get input data
#     placeholders = control_params["placeholders"]
#     x = placeholders["x"]
#     pot_points = placeholders["potential_points"]
    
    hy_param = hy.get_hyperparameter() # ハイパーパラメータの取得

    print("inference: hy_param", hy_param)
    
    # z_s: (x.shape[0] x T x dim)
    z_s, z_params = sampleVariationalDist(n_steps, init_params_flag=True, config=config)

#     init_state = get_init_dist(bs, hy_param["dim"])

#     if config["dynamics_type"] == "function": # functionの場合
#         z_pred_s, z_pred_params = sampleTransitionFromDist(
#             z_params,
#             n_steps,
#             init_state,
#             init_params_flag=True,
#             control_params=control_params,
#         )
#     elif config["dynamics_type"] == "distribution": # distributionの場合
#         z_pred_s, z_pred_params = sampleTransition(
#             z_s,
#             n_steps,
#             init_state,
#             init_params_flag=True,
#             control_params=control_params,
#         )

#     pot_loss = None
#     # compute emission
#     obs_params = computeEmission(
#         z_s, n_steps, init_params_flag=True, control_params=control_params
#     )
#     obs_pred_params = computeEmission(
#         z_pred_s, n_steps, init_params_flag=False, control_params=control_params
#     )

#     pot_loss=computePotentialLoss(z_params, pot_points, n_steps, control_params=control_params)
    
#     outputs = {
#         "z_s": z_s,
#         "z_params": z_params,
#         "z_pred_s": z_pred_s,
#         "z_pred_params": z_pred_params,
#         "obs_params": obs_params,
#         "obs_pred_params": obs_pred_params,
#         "potential_loss": pot_loss,
#     }
    
#     return outputs



# def inference(n_steps, config):
#     """
#     2種類のモードがある
    
#     出力
#     -------
#     inference results by sampling or function.
#     """
    
#     # デフォルトでconfig["dynamics_type"]=="distribution"
    
#     if config["dynamics_type"] == "distribution": # 
#         return inference_by_sample(n_steps, config)
#     elif config["dynamics_type"] == "function": # 
#         return inference_by_dist(n_steps, config)
#     else: # condif["dynamics_type"]が適切でない場合、エラーを吐く
#         raise Exception("[Error] unknown dynamics type")


def train(config): # 学習、main()で実行
    hy_param = hy.get_hyperparameter() # ハイパーパラメータを取得する
    
    print("train: hy_param", hy_param)
    
    train_data, valid_data = dmm_input.load_data(config,with_shuffle=True,with_train_test=True) # データをロードする
    
    batch_size, n_batch = get_batch_size(config, train_data)
    dim, dim_emit = get_dim(config, hy_param, train_data)
    
    n_steps = train_data.n_steps # 時系列長？
    
    
    
    hy_param["n_steps"] = n_steps # hy_paramにはデフォルトではない
    
    print("train(): train_data_size:", train_data.num) # sampleだとtrain_data_size: 2
    print("train(): batch_size     :", batch_size) # sampleだとbatch_size     : 2
    print("train(): n_steps        :", n_steps) # sampleだとn_steps        : 3
    print("train(): dim_emit       :", dim_emit) # sampleだとdim_emit       : 4

    
    
    # configにもともとはplaceholderも入っていた
    outputs = inference(n_steps, config=config)
    
    
    
    
    
    
    
    
    
