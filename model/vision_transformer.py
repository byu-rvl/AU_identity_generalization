import torch
import torch.nn as nn


class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x

def load_vit_b_16(pretrained=True):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.DEFAULT
    
    model = vit_b_16(weights=weights)
    ouput_size = model.heads.head.in_features
    model.heads = Identity()

    
    return model, ouput_size

def load_vit_b_32(pretrained=True):
    from torchvision.models import vit_b_32, ViT_B_32_Weights
    weights = ViT_B_32_Weights.DEFAULT
    model = vit_b_32(weights=weights)
    ouput_size = model.heads.head.in_features
    model.heads = Identity()
    
    return model, ouput_size

def load_vit_l_16(pretrained=True):
    from torchvision.models import vit_l_16, ViT_L_16_Weights
    weights = ViT_L_16_Weights.DEFAULT
    print("weights", weights)
    model = vit_l_16(weights=weights)
    ouput_size = model.heads.head.in_features
    model.heads = Identity()
    
    return model, ouput_size

def load_vit_l_32(pretrained=True):
    from torchvision.models import vit_l_32, ViT_L_32_Weights
    weights = ViT_L_32_Weights.DEFAULT
    print("weights", weights)
    model = vit_l_32(weights=weights)
    ouput_size = model.heads.head.in_features
    model.heads = Identity()
    
    return model, ouput_size

# needs larger 518 input. Implement later?
# def load_vit_h_14(pretrained=True):
#     from torchvision.models import vit_h_14, ViT_H_14_Weights
#     weights = ViT_H_14_Weights.DEFAULT
#     print("weights", weights)
#     model = vit_h_14(weights=weights)
#     ouput_size = model.heads.head.in_features
#     model.heads = Identity()
    
#     return model, ouput_size

def load_inception_v3(pretrained=True):
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=pretrained)
    ouput_size = model.fc.in_features
    model.fc = Identity()

    return model, ouput_size

if __name__=="__main__":
    input = torch.randn(10,3,299,299)
    print("Downloading models...")
    model, model_size = load_inception_v3()
    print(model)
    model.fc = nn.Identity()  # Replace it with an identity layer
    # model.Mixed_7c.branch_pool.bn = nn.Identity()  # Replace it with an identity layer
    # model.Mixed_7c.branch3x3dbl_3b = nn.Identity()  # Replace it with an identity layer
    # model.avgpool = nn.Identity()  # Replace it with an identity layer
    # model.dropout = nn.Identity()  # Replace it with an identity layer
    # print(model)

    out = model(input)
    print(out.logits.shape)
    # print(out.shape)
    print(model_size)
    # try:
    #     load_vit_b_16()
    # except Exception as e:
    #     pass
    # try:
    #     load_vit_b_32()
    # except Exception as e:
    #     pass
    # try:
    #     load_vit_l_16()
    # except Exception as e:
    #     pass
    # try:
    #     load_vit_l_32()
    # except Exception as e:
    #     pass
    print("Done")
