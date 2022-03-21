import omegaconf
from data.make_dataset import main as get_data
from omegaconf import OmegaConf

def pipeline(params, config_path = 'configs/default.yaml'):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    models : _type_
        _description_
    params : _type_
        _description_

    Returns
    -------
    results: _type_
        _description_
    """    
    configs = OmegaConf.load(config_path)
    Y, params = get_data(configs.data)
    Phi = Y[1] if params.type == 'stationary_smooth' else []
    
    model_params = OmegaConf.load(configs.model)

    L, S, est_Phi, obj_val, lam_val = simglemare(Y, model_params)


    return results