import glob
from pathlib import Path
import tqdm


def create_labels(EBplus_dir, save_path):

    all_csvs = list(glob.glob(f"{EBplus_dir}/FACS_occ/*.csv"))
    print(f"Found {len(all_csvs)} CSV files for EB+ dataset.")

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # EB AUS: 1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,27,28,29,30,31,32,33,34,35,36,37,38,39,99
    # BP4D/DISFA AUs: 1,2,4,6,7,8,10,12,14,15,17,23,24,25,26
    # Overlap AUs: 1,2,4,6,7,10,12,14,15,17,23,24
    desired_aus = [1,2,4,6,7,10,12,14,15,17,23,24]

    img_paths = []
    labels = []

    for csv_file in tqdm.tqdm(all_csvs):
        # load csv file
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        subjectNumber = Path(csv_file).stem.split("_")[0]
        action = Path(csv_file).stem.split("_")[1]

        action_path = f"{subjectNumber}/{action}/"

        header = lines[0].strip().split(',')
        # remove first index which is frame number
        header = header[1:]
        header = [int(au) for au in header]

        # find indices of desired AUs
        desired_indices = [header.index(au) for au in desired_aus if au in header]

        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            frame_num = int(parts[0]) - 1 # -1 is to convert to 0-based index
            au_labels = parts[1:]  # assuming rest are AU labels
            au_labels = [au_labels[i] for i in desired_indices]

            img_file = f"{action_path}/{frame_num}.jpg"
            
            img_paths.append(img_file)
            labels.append(' '.join(au_labels))

    # Save to file
    print(f"Saving labels for {len(img_paths)} images to {save_path}/EBplus_image_labels.txt")

    with open(f"{save_path}/EBplus_image_labels.txt", 'w') as f:
        for img_path, label in zip(img_paths, labels):
            f.write(f"{label}\n")

    print(f"Saving image paths for {len(img_paths)} images to {save_path}/EBplus_image_paths.txt")

    with open(f"{save_path}/EBplus_image_paths.txt", 'w') as f:
        for img_path in img_paths:
            f.write(f"{img_path}\n")

    print("Label creation completed.")

if __name__ == "__main__":
    EBplus_dir = "/home/andreww9/groups/grp_RVL_AU/code/EBPlus"
    # save_path = "/home/andreww9/groups/grp_RVL_AU/code/EBPlus_processed_videos_only/list/"
    save_path = "/home/andreww9/groups/grp_RVL_AU/nobackup/autodelete/EBPlus_processed_videos_only/list/"
    create_labels(EBplus_dir, save_path)