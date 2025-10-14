import cv2
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from pathlib import Path
import glob
import numpy as np
from skimage import transform as trans
import tqdm

def preprocess(img, bbox=None, landmark=None, **kwargs):
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
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == image_size[1]
        assert image_size[0] % 2 == 0
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
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped

if __name__ == "__main__":

    # # all_images = list(glob.glob("/home/andreww9/groups/grp_face_race/code/VoxCeleb1_train_best_frames/id10001*.jpg"))
    # all_images = list(glob.glob("/home/andreww9/fsl_groups/grp_face_race/code/VoxCeleb1_train_best_frames_new/*.jpg"))
    # # save_path = Path("/home/andreww9/groups/grp_face_race/code/VoxCeleb1_train_best_frames_mtcnn_test/")
    # save_path = Path("/home/andreww9/fsl_groups/grp_face_race/code/VoxCeleb1_train_best_frames_mtcnn_new/")
    all_images = list(glob.glob("/home/andreww9/groups/grp_ensembleAU2/nobackup/autodelete/VoxCeleb2_train_best_frames/*.jpg"))
    save_path = Path("/home/andreww9/groups/grp_ensembleAU2/nobackup/autodelete/VoxCeleb2_train_best_frames_mtcnn_new/")
    save_path.mkdir(parents=True, exist_ok=True)

    detector = MTCNN(device="CPU:0")

    for image_path in tqdm.tqdm(all_images):

        # Load an image
        image = load_image(image_path)

        # Load face center
        face_center = np.load(image_path.replace('.jpg', '_face_center.npy'))

        # Detect faces in the image
        result = detector.detect_faces(image)

        for i in range(len(result)):
            # See if the face center is within the bounding box
            box = result[i]['box']  # [x, y, width, height]
            if face_center[0] < box[0] or face_center[0] > box[0] + box[2] or face_center[1] < box[1] or face_center[1] > box[1] + box[3]:
                continue


            keypoints_dict = result[i]["keypoints"]
            keypoints = np.array([[keypoints_dict["left_eye"][0], keypoints_dict["left_eye"][1]],
                                [keypoints_dict["right_eye"][0], keypoints_dict["right_eye"][1]],
                                [keypoints_dict["nose"][0], keypoints_dict["nose"][1]],
                                [keypoints_dict["mouth_left"][0], keypoints_dict["mouth_left"][1]],
                                [keypoints_dict["mouth_right"][0], keypoints_dict["mouth_right"][1]]])
            face = preprocess(image, result[i]["box"], keypoints, image_size="224")

            save_name = save_path / Path(image_path).name
            cv2.imwrite(str(save_name), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))