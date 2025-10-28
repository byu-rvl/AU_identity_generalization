import cv2
import glob
import tqdm
import os

if __name__ == "__main__":
    # testFiles = "/home/andreww9/fsl_groups/grp_AU_storage/compute/FEC_dataset_downloader/train_images"
    testFiles = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images"
    allImages = list(glob.glob(testFiles + "/**"))

    allNone = []
    for image in tqdm.tqdm(allImages):
        img = cv2.imread(image)
        if img is None:
            print("Image is None: ", image)
            allNone.append(image)
            # delete the image.
            os.remove(image)

    print("All None: ", allNone)
    print("Number of None: ", len(allNone))