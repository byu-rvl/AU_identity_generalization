import cv2
import glob
import imageio
from skimage.transform import resize
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torch

from third_party_helpers.access_fsrt import access_fsrt


def make_dataset(image_list, label_list, au_relation=None, landmark_list=None, train=False):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:], landmark_list[i].strip()) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

class RunFRST:
    def __init__(self, conf):
        self.fsrt_model = access_fsrt(conf)
        self.to_pil = transforms.ToPILImage()
        if conf.fsrt_vox == 1:
            self.all_sources = list(glob.glob("/home/andreww9/fsl_groups/grp_face_race/code/VoxCeleb1_train_best_frames_mtcnn_new/*.jpg"))
        elif conf.fsrt_vox == 2:
            self.all_sources = list(glob.glob("/home/andreww9/groups/grp_ensembleAU2/nobackup/autodelete/VoxCeleb2_train_best_frames_mtcnn_new/*.jpg"))
        else:
            raise Exception("fsrt_vox must be 1 or 2")
        # Shuffle the source images to ensure variety
        random.shuffle(self.all_sources)
    
    def run_fsrt(self, target_image, index):
        if index > len(self.all_sources) - 1:
            index = index % len(self.all_sources)
        source_image = resize(imageio.imread(self.all_sources[index]), (256, 256))[..., :3]
        with torch.no_grad():
            output = self.fsrt_model.run_fsrt_list(np.array([source_image]), [target_image])
        return output

class RunCLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    def run_clahe(self, target_image):
        # Ensure the image is in uint8 format
        target_image = (target_image * 255).astype(np.uint8)
        
        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Normalize to [0, 1]
        final = final.astype(np.float32) / 255.0
        return final

class FEC(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, conf=None):

        self._root_path = root_path
        self._train = train
        if train:
            self.img_folder_path = "/home/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images"
            # self.img_folder_path = "/tmp/" + str(conf.jobID) + "/train_images"
        else:
            self.img_folder_path = "/home/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images"
            # self.img_folder_path = "/tmp/" + str(conf.jobID) + "/test_images"
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        
        if self._train:
            self.allInfo_images = np.load(root_path + "/list/FEC_train_images.npy")
            self.allInfo_annotations = np.load(root_path + "/list/FEC_train_annotations.npy")
        else:
            self.allInfo_images = np.load(root_path + "/list/FEC_test_images.npy")
            self.allInfo_annotations = np.load(root_path + "/list/FEC_test_annotations.npy")
        

    def __getitem__(self, index):
        theImages = self.allInfo_images[index]
        theAnnotations = self.allInfo_annotations[index]
        theImages_out = []
        for i in range(0, 3):
            img = self.loader(os.path.join(self.img_folder_path, theImages[i]))
            # img = self.loader(os.path.join(self.img_folder_path, "0.jpeg"))
            w, h = img.size
            if h > self.crop_size:
                offset_y = random.randint(0, h - self.crop_size)
            else:
                offset_y = 0
            if w > self.crop_size:
                offset_x = random.randint(0, w - self.crop_size)
            else:
                offset_x = 0
            flip = random.randint(0, 1)
            if self._transform is not None:
                if self._train:
                    img = self._transform(img, flip, offset_x, offset_y)
                else:
                    img = self._transform(img)
            theImages_out.append(img)


        return theImages_out[0], theImages_out[1], theImages_out[2], theAnnotations

    def __len__(self):
        return len(self.allInfo_images)

class BP4D(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, conf=None):

        assert (fold>0 and fold <=3) or (fold == -1), 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        # self.lmk_folder_path = os.path.join(root_path,'Landmark_Points_np')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # img landmarks
            train_landmark_list_path = os.path.join(root_path, 'list', 'BP4D_train_lmk_path_fold' + str(fold) + '.txt')
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
            au_relation_list = np.loadtxt(au_relation_list_path)
            self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list, train=self._train)
        elif fold == -1:
            all_data_list = []
            for f in range(1,4):
                # img
                test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(f) + '.txt')
                test_image_list = open(test_image_list_path).readlines()

                # img labels
                test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(f) + '.txt')
                test_label_list = np.loadtxt(test_label_list_path)
                all_data_list.extend(make_dataset(test_image_list, test_label_list, train=self._train))
            self.data_list = all_data_list
        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list, train=self._train)
        self.to_pil = transforms.ToPILImage()
        self.preprocessing = None
        self.proportion_with_frst = conf.proportion_with_frst
        if conf.do_clahe:
            self.do_clahe = True
            self.clahe_processor = None
        else:
            self.do_clahe = False
        self.conf = conf

    def __getitem__(self, index):
        if self._train:
            img, label, au_relation, landmark_path = self.data_list[index]

            img = np.array(resize(imageio.imread(os.path.join(self.img_folder_path, img)), (256, 256))[..., :3])
            if self.do_clahe and self.clahe_processor is None:
                self.clahe_processor = RunCLAHE()
            if self.preprocessing is None and self.proportion_with_frst > 0.0:
                self.preprocessing = RunFRST(self.conf)
            if self.do_clahe:
                img = self.clahe_processor.run_clahe(img)
            if random.random() < self.proportion_with_frst:
                img = self.preprocessing.run_fsrt(img, index)[0]
                if self.do_clahe:
                    img = self.clahe_processor.run_clahe(img.numpy())
                else:
                    # When I don't have the run_clahe, I need to reshape to C, H, W
                    img = np.transpose(img, (2, 0, 1))
            img = self.to_pil(img)
            # Save image for debugging
            # img.save(f"debug/debug_img_{index}.jpg")
            if self._transform is not None: 
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                img = self._transform(img, flip, offset_x, offset_y)
            
            landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))

            return img, label, au_relation, landmark
        else:
            img, label = self.data_list[index]
            
            img = np.array(resize(imageio.imread(os.path.join(self.img_folder_path, img)), (256, 256))[..., :3])

            if self.do_clahe and self.clahe_processor is None:
                self.clahe_processor = RunCLAHE()
            if self.do_clahe:
                img = self.clahe_processor.run_clahe(img)

            if self.conf.eval_fsrt and self.proportion_with_frst > 0.0:
                if self.preprocessing is None:
                    self.preprocessing = RunFRST(self.conf)
                img = self.preprocessing.run_fsrt(img, index)[0]
                if self.do_clahe:
                    img = self.clahe_processor.run_clahe(img.numpy())
                else:
                    img = np.transpose(img, (2, 0, 1))

            img = self.to_pil(img)
            if self._transform is not None:
                img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader, conf=None):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        self.lmk_folder_path = os.path.join(root_path,'lmk')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)
            # landmarks
            train_landmark_list_path = os.path.join(root_path, 'list', 'DISFA_train_landmark_path_fold' + str(fold) + '.txt')
            train_landmark_list = open(train_landmark_list_path).readlines()

            # AU relation
            au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
            au_relation_list = np.loadtxt(au_relation_list_path)
            self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list, train_landmark_list, train=self._train)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list, train=self._train)
        self.to_pil = transforms.ToPILImage()
        self.preprocessing = None
        self.proportion_with_frst = conf.proportion_with_frst
        if conf.do_clahe:
            self.do_clahe = True
            self.clahe_processor = None
        else:
            self.do_clahe = False
        self.conf = conf
            
    def __getitem__(self, index):
        if self._train:
            img, label, au_relation, landmark_path = self.data_list[index]

            img = np.array(resize(imageio.imread(os.path.join(self.img_folder_path, img)), (256, 256))[..., :3])
            if self.do_clahe and self.clahe_processor is None:
                self.clahe_processor = RunCLAHE()
            if self.preprocessing is None and self.proportion_with_frst > 0.0:
                self.preprocessing = RunFRST(self.conf)
            if self.do_clahe:
                img = self.clahe_processor.run_clahe(img)
            if random.random() < self.proportion_with_frst:
                img = self.preprocessing.run_fsrt(img, index)[0]
                if self.do_clahe:
                    img = self.clahe_processor.run_clahe(img.numpy())
                else:
                    # When I don't have the run_clahe, I need to reshape to C, H, W
                    img = np.transpose(img, (2, 0, 1))
            img = self.to_pil(img)
            # Save image for debugging
            # img.save(f"debug/debug_img_{index}.jpg")
            if self._transform is not None: 
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                img = self._transform(img, flip, offset_x, offset_y)
            
            landmark = np.load(os.path.join(self.lmk_folder_path, landmark_path))

            return img, label, au_relation, landmark
        else:
            img, label = self.data_list[index]
            
            img = np.array(resize(imageio.imread(os.path.join(self.img_folder_path, img)), (256, 256))[..., :3])

            if self.do_clahe and self.clahe_processor is None:
                self.clahe_processor = RunCLAHE()
            if self.do_clahe:
                img = self.clahe_processor.run_clahe(img)

            if self.conf.eval_fsrt and self.proportion_with_frst > 0.0:
                if self.preprocessing is None:
                    self.preprocessing = RunFRST(self.conf)
                img = self.preprocessing.run_fsrt(img, index)[0]
                if self.do_clahe:
                    img = self.clahe_processor.run_clahe(img.numpy())
                else:
                    img = np.transpose(img, (2, 0, 1))

            img = self.to_pil(img)
            if self._transform is not None:
                img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)