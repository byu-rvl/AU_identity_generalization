import imageio
import sys
import os
import torch
import torchvision.transforms as T
import yaml
from skimage.transform import resize
import numpy as np
from skimage import img_as_ubyte
from tqdm import tqdm
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

fsrt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'fsrt'))

# 2. Add this path to the beginning of Python's search paths
if fsrt_path not in sys.path:
    sys.path.insert(0, fsrt_path)

from srt.checkpoint import Checkpoint
from modules.keypoint_detector import KPDetector
from modules.expression_encoder import ExpressionEncoder
from srt.model import FSRT
from demo import extract_keypoints_and_expression, normalize_kp, forward_model

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

        self.resize_transform = T.Resize(size=(224, 224), antialias=True) 

    def make_animation(self, source_image, driving_video, model, kp_detector, cfg, max_num_pixels, relative=False, adapt_movement_scale=False):
        _, y, x= np.meshgrid(np.zeros(2),np.arange(source_image.shape[-3]), np.arange(source_image.shape[-2]), indexing="ij")
        idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
        #Normalize
        idx_grids[...,0] = (idx_grids[...,0]+0.5 -((source_image.shape[-3])/2.0))/((source_image.shape[-3])/2.0)
        idx_grids[...,1] = (idx_grids[...,1]+0.5 -((source_image.shape[-2])/2.0))/((source_image.shape[-2])/2.0)
        idx_grids = torch.from_numpy(idx_grids).cuda().unsqueeze(0)
        z = None
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image.astype(np.float32)).permute(0, 3, 1, 2).cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source, expression_vector_src = extract_keypoints_and_expression(source.clone(), model, kp_detector,cfg, src=True)
            kp_driving_initial, _ = extract_keypoints_and_expression(driving[:, :, 0].cuda().clone(), model, kp_detector,cfg)

            for frame_idx in range(driving.shape[2]):
                driving_frame = driving[:, :, frame_idx].cuda()
                kp_driving, expression_vector_driv = extract_keypoints_and_expression(driving_frame.clone(), model, kp_detector,cfg)
                
                kp_norm = normalize_kp(kp_source=kp_source[0], kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                    adapt_movement_scale=adapt_movement_scale)
                    
                out, z =  forward_model(model,expression_vector_src, kp_source, expression_vector_driv, kp_norm, source.unsqueeze(0), idx_grids, cfg, max_num_pixels, z=z)
                #img_kp = torch.from_numpy(draw_image_with_kp(torch.clamp(out[0],0.,1.).cpu().numpy(),kp_norm['kp'][0].cpu().numpy()))
                predictions.append(torch.cat([driving_frame.detach()[0].permute(1,2,0).cpu(),torch.clamp(out[0],0.,1.)],dim=-2))
        return predictions

    
    def run_fsrt_list(self, source_image, driving_video, save=False):
        '''
        Inputs:
            source_image: np.array of np.array images of shape (256, 256, 3)
            driving_video: list of np.array images of shape (256, 256, 3)
            save: boolean of whether to save the video as res
 
        '''
        predictions = self.make_animation(source_image, driving_video, self.model, self.kp_detector, relative=self.relative, adapt_movement_scale=self.adapt_scale, cfg=self.cfg, max_num_pixels=self.max_num_pixels)
        predictions = [x[:, 256:, :] for x in predictions]

        if save:
            imageio.mimsave("result.mp4", [img_as_ubyte(frame) for frame in predictions], fps=20)

        return predictions

    def make_animation_batch(self, source_image, driving_video, model, kp_detector, cfg, max_num_pixels, relative=False, adapt_movement_scale=False):
        _, y, x= np.meshgrid(np.zeros(2),np.arange(source_image.shape[-3]), np.arange(source_image.shape[-2]), indexing="ij")
        idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
        #Normalize
        idx_grids[...,0] = (idx_grids[...,0]+0.5 -((source_image.shape[-3])/2.0))/((source_image.shape[-3])/2.0)
        idx_grids[...,1] = (idx_grids[...,1]+0.5 -((source_image.shape[-2])/2.0))/((source_image.shape[-2])/2.0)
        idx_grids = torch.from_numpy(idx_grids).cuda().unsqueeze(0)
        # Repeat idx_grids for batch size
        idx_grids = idx_grids.repeat(driving_video.shape[0], 1, 1, 1, 1)

        z = None
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image.astype(np.float32)).permute(0, 3, 1, 2).cuda()
            # driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            driving = driving_video.permute(0, 3, 1, 2).cuda()
            kp_source, expression_vector_src = extract_keypoints_and_expression(source.clone(), model, kp_detector,cfg, src=True)
            kp_driving_initial, _ = extract_keypoints_and_expression(driving.clone(), model, kp_detector,cfg)

            driving_frame = driving
            # driving_frame = driving_video.permute(0, 3, 1, 2).cuda()
            kp_driving, expression_vector_driv = extract_keypoints_and_expression(driving_frame.clone(), model, kp_detector,cfg)
            
            kp_norm = normalize_kp(kp_source=kp_source[0], kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                adapt_movement_scale=adapt_movement_scale)
            # kp_norm = kp_driving

            expression_vector_src = expression_vector_src.squeeze(1)
            expression_vector_src = expression_vector_src.repeat(driving_video.shape[0], 1, 1)

            kp_source = kp_source.repeat(driving_video.shape[0], 1, 1, 1)

            source_repeated = source.repeat(driving_video.shape[0], 1, 1, 1)
                
            out, z =  forward_model(model,expression_vector_src, kp_source, expression_vector_driv, kp_norm, source_repeated, idx_grids, cfg, max_num_pixels, z=z)
            out = out.permute(0, 3, 1, 2)
            predictions = out
        return predictions

    def run_fsrt_list_batch(self, source_images, driving_videos, save=False):
        '''
        Inputs:
            source_images: np.array of np.array images of shape (B, 256, 256, 3)--the identity images
            driving_videos: list of list of np.array images of shape (B, T, 256, 256, 3)--the expression videos
            save: boolean of whether to save the video as result
        '''
        driving_videos = driving_videos.float()
        predictions = self.make_animation_batch(np.array([source_images]), driving_videos, self.model, self.kp_detector, relative=self.relative, adapt_movement_scale=self.adapt_scale, cfg=self.cfg, max_num_pixels=self.max_num_pixels)

        predictions = [torch.clamp(self.resize_transform(x), 0, 1) for x in predictions]
        predictions = torch.stack(predictions)

        if save:
            for i, frame in enumerate(predictions):
                print("frame.shape:", frame.shape, torch.max(frame), torch.min(frame), frame.dtype)
                print("source_images.shape:", source_images.shape, np.max(source_images), np.min(source_images), source_images.dtype)
                print("driving_videos.shape:", driving_videos.shape, torch.max(driving_videos), torch.min(driving_videos), driving_videos.dtype)

                imageio.imwrite(f"output/{i}_frame.png", img_as_ubyte(frame.permute(1, 2, 0).cpu().numpy()))
                imageio.imwrite(f"output/{i}_source.png", img_as_ubyte(source_images))
                imageio.imwrite(f"output/{i}_driving.png", img_as_ubyte(driving_videos[i].cpu().numpy()))
            raise Exception("Saved frames")
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
        driving_video = driving_video[:20]
        print("Took first 20 frames")

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