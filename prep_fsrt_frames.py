from dataset import *
import argparse
from easydict import EasyDict as edict
from utils import *
from pathlib import Path
from skimage.transform import resize
import tqdm
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--datasetFSRT', type=str, help="experiment dataset BP4D / DISFA")
    parser.add_argument('--epochFSRT', type=int, help='which epoch FRST model to use')
    parser.add_argument('--jobID', type=int, help='which epoch FRST model to use')
    config, unparsed = parser.parse_known_args()
    conf = edict(config.__dict__)

    random.seed(conf.epochFSRT)

    conf.fsrt_vox = 2

    fsrt_helper = RunFRST(conf)

    print(len(fsrt_helper.all_sources))

    unique_ids = set()
    id_options = {}
    for src in fsrt_helper.all_sources:
        unique_ids.add(Path(src).stem.split('_')[0])
        if Path(src).stem.split('_')[0] not in id_options:
            id_options[Path(src).stem.split('_')[0]] = []
        id_options[Path(src).stem.split('_')[0]].append(src)
    print(f"Unique identities in FRST source images: {len(unique_ids)}")
    # print(unique_ids)
    epoch = conf.epochFSRT


    bp4d_dataset_path = "/fslhome/andreww9/code/original_datasets/BP4D_croppped_MTCNN"
    disfa_dataset_path = "/home/andreww9/fsl_groups/grp_RVL_AU/code/DISFA_LEFT_cropped_MTCNN"
    crop_size = 224
    available_ids = list(unique_ids.copy())
    available_id_options = copy.deepcopy(id_options)

    if conf.datasetFSRT == "BP4D":
        use_dataset_path = bp4d_dataset_path
        use_save_path = "/tmp/prep_fsrt/" + str(conf.jobID) + f"/BP4D_epoch{epoch}"
    elif conf.datasetFSRT == "DISFA":
        use_dataset_path = disfa_dataset_path
        use_save_path = "/tmp/prep_fsrt/" + str(conf.jobID) + f"/DISFA_epoch{epoch}"
    else:
        raise Exception("Unknown dataset for FRST preprocessing.")

    
    # for epoch in range(1,21):
    track_conversion = ""
    for fold in range(1, 4):
        img_paths = f"{use_dataset_path}/list/{conf.datasetFSRT}_test_img_path_fold{fold}.txt"
        with open(img_paths, 'r') as f:
            lines = f.readlines()
            print(f"Fold {fold}: {len(lines)} images in {conf.datasetFSRT} test set.")

        for line in tqdm.tqdm(lines):
            # print(line.strip())
            img = line.strip()
            # Randomly select a source identity 
            if len(available_ids) == 0:
                available_ids = list(unique_ids.copy())
            new_id = random.choice(available_ids)
            available_ids.remove(new_id)
            # print(f"Selected FRST source identity: {new_id}")
            if len(available_id_options[new_id]) == 0:
                available_id_options[new_id] = id_options[new_id].copy()
            source_img_path = random.choice(available_id_options[new_id])
            # print(f"Using source image: {source_img_path}")
            available_id_options[new_id].remove(source_img_path)

            target_image = np.array(resize(imageio.imread(os.path.join(f"{use_dataset_path}/img/", img)), (256, 256))[..., :3])
            source_image = resize(imageio.imread(source_img_path), (256, 256))[..., :3]
            with torch.no_grad():
                output = fsrt_helper.fsrt_model.run_fsrt_list(np.array([source_image]), [target_image])

            save_path = f"{use_save_path}/{img}"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            imageio.imsave(save_path, (output[0]*255).numpy().astype(np.uint8))

            track_conversion += f"{img}  -->  {save_path.replace(use_save_path, '')}\n"
            
    log_path = f"{use_dataset_path}/list/fsrt_conversion_log_epoch{epoch}.txt"
    with open(log_path, 'w') as f:
        f.write(track_conversion)
    print(f"FRST preprocessing completed for epoch {epoch} on dataset {conf.datasetFSRT}. Log saved to {log_path}.")