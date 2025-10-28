import pandas as pd
import numpy as np
import os
import tqdm
from pathlib import Path
import glob

if __name__ == "__main__":
    # csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/faceexp-comparison-data-train-public.xlsx"
    # save_csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/train_images_with_index.csv"
    # index_url = np.load("index_url.npy", allow_pickle=True).item()
    csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/faceexp-comparison-data-test-public.xlsx"
    save_csvFile = "/fslhome/andreww9/fsl_groups/grp_AU_storage/code/FEC_dataset_downloader/test_images_with_index.csv"
    index_url = np.load("test_index_url.npy", allow_pickle=True).item()
    

    # Read the csv file
    df = pd.read_excel(csvFile, header=None).to_numpy()

    new_csv = []
    for i in tqdm.tqdm(range(1, len(df))): # skip the first row for the header.
        image0url = df[i, 0]
        image1url = df[i, 5]
        image2url = df[i, 10]
        
        image0index = index_url[image0url]
        image1index = index_url[image1url]
        image2index = index_url[image2url]

        #make the first three columsn the indexes, with the rest of the column shifted over.
        new_csv.append([image0index, image1index, image2index] + list(df[i]))
    new_csv = pd.DataFrame(new_csv)
    new_csv.to_csv(save_csvFile, index=False, header=False)
