import torch as tc
from torch.linalg import pinv


def get_input(inputs, step, n_steps):
    if inputs is not None:
        input_for_step = inputs[step:(-n_steps + step), :]
    else:
        input_for_step = None
    return input_for_step


def get_ahead_pred_obs(model, data, inputs, n_steps, prewarm, prewarm_alpha):
    # dims
    T, dx = data.size()
    
    # if inputs is not None:
    #     # if len(inputs.shape)==2:
    #     #     inputs = inputs.unsqueeze(0)
    #     # inputs_ = inputs.permute(1, 0, 2)
    #     inputs_ = inputs
    # else:
    #     inputs_ = [None]*T

    # true data
    time_steps = T - n_steps - prewarm
    x_data = data[:time_steps, :].to(model.device)

    # latent model
    lat = model.latent_model

    # initial state. Uses x0 of the dataset
    # batch dim of z holds all T - n_steps time steps
    if model.z0_model:
        z = model.z0_model(x_data)
    else:
        dz = lat.d_z
        z = tc.randn((time_steps, dz), device=model.device)

        # obs. model inv?
        inv_obs = model.args['tf_mode']=='inversion'
        B_PI = None
        if inv_obs:
            B = model.output_layer.weight
            B_PI = pinv(B)
        z = lat.teacher_force(z, x_data, B_PI)

    X_pred = tc.empty((n_steps, time_steps, dx), device=model.device)
    params = model.get_latent_parameters()
    
    if inputs is not None:          
        for step in range(prewarm):
            z = lat.teacher_force_filtered(z, x_data, None, alpha=prewarm_alpha)
            z = lat.latent_step(z, *params, inputs[step:step+time_steps])
        
        for step in range(n_steps):
            # latent step performs ahead prediction on every
            # time step here
            z = lat.latent_step(z, *params, inputs[step+prewarm:step+prewarm+time_steps])
            x = model.output_layer(z)            
            X_pred[step] = x
    else:
        for step in range(prewarm):
            z = lat.teacher_force_filtered(z, x_data, None, alpha=prewarm_alpha)
            z = lat.latent_step(z, *params, None)
        
        for step in range(n_steps):
            # latent step performs ahead prediction on every
            # time step here
            z = lat.latent_step(z, *params, None)
            x = model.output_layer(z)            
            X_pred[step] = x

    return X_pred

# def get_ahead_pred_from_first_data_point(model, data, inputs):
    
#     if len(data.shape)==2:
#         data = data.unsqueeze(0)
#     x0 = data[:,0,:]
#     T = data.shape[1] - 1
    
#     if inputs is not None:
#         if len(inputs.shape)==2:
#             inputs = inputs.unsqueeze(0)
#         inputs_ = inputs.permute(1, 0, 2)
#     else:
#         inputs_ = [None]*T
    
#     if model.z0_model:
#         z = model.z0_model(x0)
#     else:
#         dz = model.latent_model.d_z
#         z = tc.randn((1, dz), device=model.device)
#         B_PI = None
#         if model.args['use_inv_tf']:
#             B = model.output_layer.weight
#             B_PI = pinv(B)
#         z = model.latent_model.teacher_force(z, x0, B_PI)
    
#     dx = data.shape[2]
#     X_pred = tc.empty((1, T, dx))
#     params = model.get_latent_parameters()
#     for t in range(T):
#         z = model.latent_model.latent_step(z, *params, inputs_[t])
#         x = model.output_layer(z)
#         X_pred[:,t] = x
        
    # return X_pred
        

def construct_ground_truth(data, n_steps):
    T, dx = data.size()
    time_steps = T - n_steps
    X_true = tc.empty((n_steps, time_steps, dx))
    for step in range(1, n_steps + 1):
        X_true[step - 1] = data[step : time_steps + step]
    return X_true


def squared_error(x_pred, x_true):
    return tc.pow(x_pred - x_true, 2)

@tc.no_grad()
def n_steps_ahead_pred_mse(model, data, inputs, n_steps, from_step=None, feature_mean=True,
                           invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
    x_pred = get_ahead_pred_obs(model, data, inputs, n_steps, prewarm['steps'], prewarm['alpha'])
    x_true = construct_ground_truth(data, n_steps).to(model.device)[:, prewarm['steps']:]
    if invert_preprocessing:
        x_pred = model.dataset.timeseries['emas'].train_preprocessor.inverse(x_pred)
        x_true = model.dataset.timeseries['emas'].train_preprocessor.inverse(x_true)
    se = squared_error(x_pred, x_true).cpu().numpy()
    if feature_mean:
        se = se.mean(2)
    if from_step is None:
        se = se.mean(1)
    else:
        se = se[:, from_step]
    return se

# @tc.no_grad()
# def last_n_steps_ahead_pred_mse(model, data, inputs, n_steps):
#     x_pred = get_ahead_pred_obs(model, data, inputs, n_steps)[:, -1, :].squeeze()
#     x_true = data[-n_steps:]
#     mse = squared_error(x_pred, x_true).mean(1).cpu().numpy()
#     return mse

# @tc.no_grad()
# def ahead_predict_from_z0_mse(model, data, inputs, feature_mean=True):
#     x_pred = get_ahead_pred_from_first_data_point(model, data, inputs).squeeze()
#     x_true = data[1:]
#     if feature_mean:
#         mse = squared_error(x_pred, x_true).mean(1).cpu().numpy()
#     else:
#         mse = squared_error(x_pred, x_true).cpu().numpy()
#     return mse

@tc.no_grad()
def const_prediction_mse(model, data, constant_predictor, n_steps, feature_mean=True,
                         invert_preprocessing=False, prewarm={'steps':0, 'alpha':0}):
    x_true = data[prewarm['steps']+1:prewarm['steps']+n_steps+1]
    x_pred = constant_predictor.unsqueeze(0).repeat((n_steps, 1))
    if invert_preprocessing:
        x_pred = model.dataset.timeseries['emas'].train_preprocessor.inverse(x_pred)
        x_true = model.dataset.timeseries['emas'].train_preprocessor.inverse(x_true)
    mse = squared_error(x_pred, x_true).cpu().numpy()
    if feature_mean:
        mse = mse.mean(axis=1)
    return mse
