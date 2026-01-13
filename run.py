import os
import sys
import argparse
import cv2
from dataset import default_loader
import imageio
import numpy as np
from skimage.transform import resize
from skimage import transform as trans
import torch
import torchvision.transforms as transforms
from PIL import Image

from model.encoder_gcn import MEFARG
from utils import image_test, load_state_dict
from conf import get_config

# import mxnet as mx
from facenet_pytorch import MTCNN

# Function modified from file: AU_identity_generalization/preprocessing/Face_crop_align_mtcnn/align_mtcnn/mtcnn_detector.py
def preprocess(img, bbox=None, landmark=None, rotateScalePoints=None, image_size=None, **kwargs):
    """
    crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
    crop_imgs: list, n
        cropped and aligned faces
    """
    M = None
    # image_size = []
    # if image_size is not None:
    #     if len(image_size) == 1:
    #         image_size = [image_size[0], image_size[0]]
    #     assert len(image_size) == 2
    #     assert image_size[0] == image_size[1]
    #     assert image_size[0] % 2 == 0
    if landmark is not None:
        assert len(image_size) == 2
        # 这个基准是112*96的面部特征点的坐标
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        # if image_size[0] != 112:
        #     src[:, 1] += (image_size[0] - 112)/2
        # if image_size[1] != 96:
        #     src[:, 0] += (image_size[1] - 96)/2
        # if image_size[1] != 112:
        src[:, 0] += 8
        # make the mark points up 8 pixels, for crop the chin in the cropped image
        src[:, 1] -= 8

        if image_size[0] == image_size[1] and image_size[0] != 112:
            src = src/112*image_size[0]

        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    # Use perspectiveTransform to map the points
    new_lmk_points = []
    if rotateScalePoints is not None:
        for point in rotateScalePoints:
            newpoint = cv2.perspectiveTransform(np.array([[[point[0], point[1]]]], dtype=np.float32), tform.params)
            # print(newpoint)
            new_lmk_points.append([newpoint[0][0][0]/image_size[1], newpoint[0][0][1]/image_size[1]])
            # print("newpoint:", newpoint[0][0][0]/dst_width, newpoint[0][0][1]/dst_height)
        new_lmk_points = np.array(new_lmk_points)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        print("det:", det)
        print("margin:", margin)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret, new_lmk_points
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        return warped, new_lmk_points

def load_model(conf, model_path):
    conf.agg = "original"
    net = MEFARG(num_classes=15, backbone="swin_transformer_base", numEncoderLayers=3, numLandmarks=66, conf=conf)
    
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No checkpoint found at '{model_path}'")
        
    net = load_state_dict(net, model_path)
    net.eval()
    return net

def load_image(image_path, run_crop):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        raise FileNotFoundError(f"Image file not found at {image_path}")

    if run_crop:
        # NOTE: For training and evaluations in the paper, we used MxNet MTCNN implementation.
        # Here we use facenet-pytorch MTCNN for simplicity.

        #initialize mtcnn
        mtcnn = MTCNN(
            keep_all=True, 
            device="cpu",
            post_process=False  # Returns raw boxes/landmarks like your old detector
        )

        face_img = cv2.imread(image_path)
        # ret = mtcnn.detect_face(face_img)
        # CONVERT to RGB for facenet-pytorch
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)

        if boxes.shape[0] >= 1:
            bbox_ = boxes[0:1]
        else:
            raise Exception("Face not found in image. (We use the previous frame in our paper when the face was not found.)")

        points_ = None #points[i, :].reshape((2, 5)).T
        face, new_lmk_points = preprocess(face_img, bbox_[0], points_, rotateScalePoints=None, image_size=[256,256])

        # Hack for running the cropping in this function.
        cv2.imwrite("tmp.png", face)
        image_path = "tmp.png"

    loader = default_loader
    transform = image_test(crop_size=224)

    image = loader(image_path)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    if run_crop:
        os.remove("tmp.png")

    return input_tensor        

def main():
    # 1. Parse arguments specifically for this script
    parser = argparse.ArgumentParser(description='Run AU detection on a single image')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--crop', action='store_true', help='Whether to run face cropping using MTCNN')
    
    # 2. Separate run.py specific args from potential config args
    args, remaining_args = parser.parse_known_args()
    
    # 3. Patch sys.argv so get_config() receives the remaining configuration arguments
    sys.argv = [sys.argv[0]] + remaining_args
    conf = get_config()
    
    # 4. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 5. Load model
    try:
        net = load_model(conf, args.checkpoint)
        net = net.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 6. Preprocess Image
    input_tensor = load_image(args.img, args.crop)
    input_tensor = input_tensor.to(device)

    # 7. Run Inference
    print(f"Running inference on {args.img}...")
    with torch.no_grad():
        outputs, _, _, _ = net(input_tensor)
    
    print("\n" + "="*45)
    print(f" Predictions")
    print("="*45)
    print(f"{'AU Index':<10} | {'Probability':<12} | {'State':<10}")
    print("-" * 45)
    
    for i, p in enumerate(outputs[0].cpu().numpy()):
        state = "Active" if p > 0.5 else "Inactive"
        print(f"{i:<10} | {p:.4f}       | {state:<10}")
    print("="*45 + "\n")

if __name__ == '__main__':
    main()
