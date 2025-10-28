import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import glob
import sys

if __name__ == "__main__":
    # imagePaths = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train/"
    # csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images_with_index.csv"
    # csvFile2 = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/faceexp-comparison-data-train-public.xlsx"
    # savePath = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images/"
    # indexURL_path = "index_url.npy"
    imagePaths = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test/"
    csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images_with_index.csv"
    csvFile2 = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/faceexp-comparison-data-test-public.xlsx"
    savePath = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images/"
    indexURL_path = "test_index_url.npy"

    if not Path(indexURL_path).exists():
        # Read the csv file
        # df = pd.read_csv(csvFile, header=None).to_numpy()
        df2 = pd.read_excel(csvFile2, header=None).to_numpy()

        allImages = list(glob.glob(imagePaths + "**"))

        # print("df shape: ", df.shape)
        # print("df: ", df[0])
        print("allImages: ", allImages[0])
        print("len(allImages): ", len(allImages))
        print("\n")
        print("df2 shape: ", df2.shape)
        print("df2: ", df2[0])
        print("df2: ", df2[1])
        print("\n")
        print("df2[0,1]: ", df2[1,1])
        print("df2[0,2]: ", df2[1,2])
        print("df2[0,3]: ", df2[1,3])
        print("df2[0,4]: ", df2[1,4])
        print("df2[0,5]: ", df2[1,5])
        print("df2[0,6]: ", df2[1,6])
        print("df2[0,7]: ", df2[1,7])
        print("df2[0,8]: ", df2[1,8])
        print("df2[0,9]: ", df2[1,9])
        print("df2[0,10]: ", df2[1,10])
        print("df2[0,11]: ", df2[1,11])
        print("df2[0,12]: ", df2[1,12])
        print("\n")
        # print("df[0,3]: ", df[0,3])

        index_url = {}
        cntr = 0
        indexes = [0, 5, 10]
        for i in tqdm.tqdm(range(1, len(df2))):
            for j in range(3):
                thisURL = df2[i, indexes[j]]
                if thisURL not in index_url:
                    index_url[thisURL] = cntr
                    cntr += 1
        # save the index_url dictionary
        np.save(indexURL_path, index_url)
    else:
        # get start index from the termianl argument
        args = sys.argv
        if len(args) == 1:
            start_index = 1
        elif len(args) != 2:
            raise Exception("len(args) != 2")
        else:
            start_index = int(args[1])


        index_url = np.load(indexURL_path, allow_pickle=True).item()
        print("len(index_url): ", len(index_url))
        print("index_url loaded.")
        Path(savePath).mkdir(parents=True, exist_ok=True)

        df2 = pd.read_excel(csvFile2, header=None).to_numpy()

        allImages = list(glob.glob(imagePaths + "**"))
        tmp = []
        for image in tqdm.tqdm(allImages):
            tmp.append(Path(image).name)
        allImages = tmp
        print("allImages: ", allImages[0])

        indexes = [0, 5, 10]
        # for i in tqdm.tqdm(range(1, len(df2))):
        for i in tqdm.tqdm(range(start_index, len(df2))):
            for j in range(3):
                thisURL = df2[i, indexes[j]]
                cntr = index_url[thisURL]
                saveHere = savePath + "/" + str(cntr) + ".jpg"
                output_name = (str(i+1)).zfill(6) + "_" + str(j+1) + ".jpeg"
                if Path(saveHere).exists():
                    continue
                if output_name not in allImages:
                    print("cntr: ", cntr)
                    continue
                
                
                fromPath = imagePaths + "/" + output_name
                if not Path(fromPath).exists():
                    raise Exception("Path does not exist: ", fromPath)
                # copy the image
                os.system("cp " + fromPath + " " + saveHere)
