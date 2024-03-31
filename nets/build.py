from .swin_transformer import SwinTransformer
import torch
import torch.nn as nn
def build_model():

    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=5000,
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.1,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=True)
    
    checkpoint = torch.load('/home/xyz/source_codes/STAM_GitHub/swin_tiny_patch4_window7_224.pth', map_location='cpu')
    if checkpoint['model']['head.weight'].shape[0]==1000: 
        checkpoint['model']['head.weight']=torch.nn.Parameter( 
            torch.nn.init.xavier_uniform(torch.empty(5000,768))) 
        checkpoint['model']['head.bias']=torch.nn.Parameter(torch.randn(5000)) 
    msg=model.load_state_dict(checkpoint['model'],strict=False)

    return model




