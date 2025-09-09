import imageio
import sys
import os
import torch
import yaml
from skimage.transform import resize
import numpy as np
from skimage import img_as_ubyte
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

fsrt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'fsrt'))

# 2. Add this path to the beginning of Python's search paths
if fsrt_path not in sys.path:
    sys.path.insert(0, fsrt_path)

from srt.checkpoint import Checkpoint
from modules.keypoint_detector import KPDetector
from modules.expression_encoder import ExpressionEncoder
from srt.model import FSRT
from demo import make_animation

class access_fsrt:
    def __init__(self):
        kp_weights_path = "weights/kp_detector.pt"
        cfg_path = "third_party/fsrt/runs/vox256/vox256.yaml"
        checkpoint_weights_path = "weights/vox256.pt"
        self.relative = True
        self.adapt_scale = True
        self.max_num_pixels = 65536

        with open(cfg_path, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.CLoader)

        self.kp_detector = KPDetector().cuda()
        self.kp_detector.load_state_dict(torch.load('weights/kp_detector.pt'))
        expression_encoder = ExpressionEncoder(expression_size=self.cfg['model']['expression_size'], in_channels=self.kp_detector.predictor.out_filters) 

        self.model = FSRT(self.cfg['model'],expression_encoder=expression_encoder).cuda()
    
        self.model.eval()
        self.kp_detector.eval()
        
        encoder_module = self.model.encoder
        decoder_module = self.model.decoder
        expression_encoder_module = self.model.expression_encoder

        #Load the checkpoints
        checkpoint = Checkpoint('./', device='cuda:0', encoder=encoder_module,
                                    decoder=decoder_module, expression_encoder=expression_encoder_module)
        load_dict = checkpoint.load(checkpoint_weights_path)

    def run_fsrt_list(self, source_image, driving_video, save=False):
        '''
        Inputs:
            source_image: list of np.array images of shape (256, 256, 3)
            driving_video: list of np.array images of shape (256, 256, 3)
            save: boolean of whether to save the video as res
 
        '''
        predictions = make_animation(source_image, driving_video, self.model, self.kp_detector, relative=self.relative, adapt_movement_scale=self.adapt_scale, cfg=self.cfg, max_num_pixels=self.max_num_pixels)
        predictions = [x[:, 256:, :] for x in predictions]

        if save:
            imageio.mimsave("result.mp4", [img_as_ubyte(frame) for frame in predictions], fps=20)

        return predictions

    def run_fsrt_image_video(self, source_image_path, driving_video_path):
        source_image = [imageio.imread(source_image_path)]
        reader = imageio.get_reader(driving_video_path)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        #TODO remove this:
        # driving_video = driving_video[:20]
        # print("Took first 20 frames")

        source_image = [resize(img, (256, 256))[..., :3] for img in source_image]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        source_image = np.array(source_image)

        return self.run_fsrt_list(source_image, driving_video)

if __name__ == "__main__":
    fsrt = access_fsrt()
    source_image = "/home/andreww9/groups/grp_face_race/code/vox_celeb_identities/id00111_frame.jpg"
    driving_video = "/home/andreww9/groups/grp_RVL_AU/code/DISFA_/Video_RightCamera/RightVideoSN001_Comp.avi"
    fsrt.run_fsrt_image_video(source_image, driving_video)

    print("hello world")