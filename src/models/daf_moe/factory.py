import torch
import torch.nn as nn

from .daf_moe.daf_moe_transformer import DAFMoETransformer

def create_model(config):

    model_name = config.model_name.lower()
    print(f"🏭 Model Factory: Building '{model_name}'...")

    if model_name == 'daf_moe':
        return DAFMoETransformer(config)
    
    elif model_name == 'tabnet':
        raise NotImplementedError("TabNet is not implemented yet.")
        
    elif model_name == 'ft_transformer':
        raise NotImplementedError("FT-Transformer is not implemented yet.")
        
    else:
        raise ValueError(f"🚨 Unknown model name: {model_name}")