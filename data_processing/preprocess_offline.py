import os
import torch
import yaml
from tqdm import tqdm
from pathlib import Path

from pusht_dset import load_pusht_slice_train_val
from transforms import make_transforms

def preprocess_and_save():
    # 1. loading configurations
    with open("/home/lizijia/Gen-JEPA/data_processing/config/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)
    
    # extract relevant config sections
    model_kwargs = full_cfg['model_kwargs']
    data_cfg = model_kwargs['data']
    custom_cfg = data_cfg['custom']
    aug_cfg = model_kwargs['data_aug']
    
    # 2. setting paths
    RAW_DATA_PATH = "/home/lizijia/JEPA_comparison/jepa-wms/datasets/pusht/pusht_noise"
    SAVE_PATH = "./processed_offline_data"
    
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 3. create transforms
    print("🎨 creating transforms...")
    transform = make_transforms(
        img_size=data_cfg['img_size'],
        auto_augment=aug_cfg['auto_augment'],
        normalize=aug_cfg['normalize']
    )

    # 4. load dataset
    print("🚚 loading dataset...")
    datasets, _ = load_pusht_slice_train_val(
        transform=transform,
        data_path=RAW_DATA_PATH,
        n_rollout=None,                       # load all data
        normalize_action=custom_cfg['normalize_action'],
        num_hist=custom_cfg['num_hist'],      # 3
        num_pred=custom_cfg['num_pred'],      # 1
        frameskip=custom_cfg['frameskip'],    # 5
        action_skip=custom_cfg['action_skip'] # 1
    )

    # 5. process and save
    print("💾 processing and saving data...")
    SAVE_STRIDE = 5 # save every 5th sample to reduce size
    for split in ["train", "valid"]:
        current_dataset = datasets[split]
        split_dir = os.path.join(SAVE_PATH, split)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"🔄 processing {split} set, total {len(current_dataset)} samples...")
        
        # iterate and save each sample
        for i in tqdm(range(0, len(current_dataset), SAVE_STRIDE)):
            # obs: {'visual': [T, 3, 224, 224], 'proprio': [T, D]}
            # act: [T, A], state: [T, S]
            obs, act, state, info = current_dataset[i] 

        # create a dict to save
        sample = {
            "visual": obs["visual"].half(), # save as half precision
            "proprio": obs["proprio"].half(),
            "action": act.half(),
            "state": state.half()
        }
        # save to disk
        torch.save(sample, os.path.join(split_dir, f"{i:06d}.pth"))

    print(f"✅ preprocessing complete, save to: {SAVE_PATH}")

if __name__ == "__main__":
    preprocess_and_save()