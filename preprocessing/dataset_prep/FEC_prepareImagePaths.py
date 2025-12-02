import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import glob

if __name__ == "__main__":
    # imagePaths = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images/"
    # csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images_with_index.csv"
    # saveName = "FEC_train"
    # index_url_path = "index_url.npy"
    imagePaths = "/home/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images/"
    csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images_with_index.csv"
    saveName = "FEC_test"
    index_url_path = "test_index_url.npy"

    index_url = np.load(index_url_path, allow_pickle=True).item()
    saveDir = "/home/andreww9/code/original_datasets/FEC_dataset2/list/"
    
    Path(saveDir).mkdir(parents=True, exist_ok=True)

    # Read the csv file
    df = pd.read_csv(csvFile, header=None).to_numpy()

    compileImages = []
    compileAnnotations = []

    for i in tqdm.tqdm(range(0, len(df))):
        image0index = str(df[i, 0])
        image0url = df[i, 3]
        image1index = str(df[i, 1])
        image1url = df[i, 8]
        image2index = str(df[i, 2])
        image2url = df[i, 13]
        if index_url[image0url] != int(image0index):
            raise Exception("index_url[image0url] != int(image0index): ", index_url[image0url], int(image0index))
        if index_url[image1url] != int(image1index):
            raise Exception("index_url[image1url] != int(image1index): ", index_url[image1url], int(image1index))
        if index_url[image2url] != int(image2index):
            raise Exception("index_url[image2url] != int(image2index): ", index_url[image2url], int(image2index))

    availableImages = list(glob.glob(imagePaths + "*.jpg"))

    print("len(df): ", len(df))
    for i in tqdm.tqdm(range(0, len(df))):
        image0index = str(df[i, 0]) + ".jpg"
        image1index = str(df[i, 1]) + ".jpg"
        image2index = str(df[i, 2]) + ".jpg"

        annotation1 = df[i, 20]
        annotation2 = df[i, 22]
        annotation3 = df[i, 24]
        annotation4 = df[i, 26]
        annotation5 = df[i, 28]
        annotation6 = df[i, 30]

        cntr = {}
        allAnnotations = [annotation1, annotation2, annotation3, annotation4, annotation5, annotation6]
        for annotation in allAnnotations:
            if annotation in cntr:
                cntr[annotation] += 1
            else:
                cntr[annotation] = 1
        maxChoice = None
        for key in cntr:
            if maxChoice is None:
                maxChoice = key
            elif cntr[key] > cntr[maxChoice]:
                maxChoice = key

        allImages = [image0index, image1index, image2index]
        ok = True
        for image in allImages:
            image = imagePaths + image
            if image not in availableImages:
                ok = False
                if Path(image).exists():
                    print("Image exists but not in availableImages: ", image)
                    raise Exception("Image exists but not in availableImages: ", image)
                break
        if ok:
            compileImages.append(allImages)
            compileAnnotations.append(maxChoice)
    
    compileImages = np.array(compileImages)
    compileAnnotations = np.array(compileAnnotations)
    print("compileImages.shape: ", compileImages.shape)
    print("compileAnnotations.shape: ", compileAnnotations.shape)

    np.save(saveDir + "/" + saveName + "_images.npy", compileImages)
    np.save(saveDir + "/" + saveName + "_annotations.npy", compileAnnotations)