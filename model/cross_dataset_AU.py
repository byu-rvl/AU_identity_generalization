import os
import sys
import torch
import yaml

fsrt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'fsrt'))

# 2. Add this path to the beginning of Python's search paths
if fsrt_path not in sys.path:
    sys.path.insert(0, fsrt_path)

from modules.keypoint_detector import KPDetector
from modules.expression_encoder import ExpressionEncoder

class cross_dataset_AU(torch.nn.Module):
    def __init__(self, num_classes):
        super(cross_dataset_AU, self).__init__()
        config = "third_party/fsrt/runs/vox256/vox256.yaml"
        checkpoint = "weights/vox256.pt"
        # config = "third_party/fsrt/runs/vox256_2Source/vox256_2Source.yaml"
        # checkpoint = "weights/vox256_2Source.pt"

        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.CLoader)

        state_dict = torch.load(checkpoint, map_location='cuda:0')

        self.kp_detector = KPDetector().cuda()
        self.kp_detector.load_state_dict(torch.load('./third_party/fsrt/fsrt_checkpoints/kp_detector.pt'))
        self.expression_encoder = ExpressionEncoder(expression_size=cfg['model']['expression_size'], in_channels=self.kp_detector.predictor.out_filters).cuda()
        print("Loading expression_encoder")
        if 'expression_encoder' in state_dict:
            self.expression_encoder.load_state_dict(state_dict['expression_encoder'])
        else:
            raise ValueError("Key 'expression_encoder' not found in checkpoint. Failed to load weights.")

        num_kps = 10
        kps_dim = 2
        self.linear1 = torch.nn.Linear(256+num_kps*kps_dim, 256 * 2).cuda()
        # self.linear1 = torch.nn.Linear(256, 256 * 2).cuda()
        self.linear2 = torch.nn.Linear(256 * 2, num_classes).cuda()
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):

        bs, c, h, w = x.shape
        nkp = self.kp_detector.num_kp
        with torch.no_grad():
            kps, latent_dict = self.kp_detector(x)
            heatmaps = latent_dict['heatmap'].view(bs,nkp,latent_dict['heatmap'].shape[-2],latent_dict['heatmap'].shape[-1])
            feature_maps = latent_dict['feature_map'].view(bs,latent_dict['feature_map'].shape[-3],latent_dict['feature_map'].shape[-2],latent_dict['feature_map'].shape[-1])
            
        # if kps.shape[1] == 1:
        #     kps = kps.squeeze(1)
            expression_vector = self.expression_encoder(feature_maps,heatmaps)

        # KPS to [B, -1].
        kps = kps.view(bs, -1)
        expression_vector = torch.concat((kps, expression_vector), dim=1)

        expression_vector = torch.relu(self.linear1(expression_vector))
        expression_vector = self.linear2(expression_vector)
        expression_vector = torch.sigmoid(expression_vector)

        # return kps, expression_vector
        return expression_vector


if __name__ == "__main__":
    # kp_detector, expression_encoder, cfg = load_model()

    image_path = "/home/andreww9/code/original_datasets/BP4D_croppped_MTCNN/img/F001/T1/0.jpg"
    from PIL import Image
    import torchvision.transforms as transforms
    to_tensor = transforms.ToTensor()
    img = Image.open(image_path).convert('RGB')
    img = to_tensor(img).unsqueeze(0).cuda()
    print("img", img.shape, img.dtype)
    
    # Make img a batch of 16. Just for testing so repeat it.
    img = img.repeat(16,1,1,1)
    print("img", img.shape, img.dtype)

    num_classes = 12
    cd_AU = cross_dataset_AU(num_classes)
    expression_vector = cd_AU(img)
    print("expression_vector", expression_vector.shape, expression_vector.dtype)
    print(expression_vector[0])
    # kps, expression_vector = cd_AU.forward(img)
    # print(kps.shape, expression_vector.shape)
    # print(expression_vector[0])