import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
sys.path.append(os.getcwd())
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import pycocotools.mask as maskUtils
#
from evaluation.groundedsam2.grounded_sam2 import Tracker
from evaluation.common_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')

    # model
    parser.add_argument(
        "--gd_config",
        default="grounded_sam2/configs/grounded_sam2_vitb.py",
        help="Grounding DINO model config file path",
    )
    parser.add_argument(
        "--gd_ckpt",
        default="download_models/GroundedDino/groundingdino_swint_ogc.pth",
        help="Grounding DINO model checkpoint file path",
    )
    parser.add_argument(
        "--sam2_config",
        default="grounded_sam2/configs/sam2/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config file path",
    )
    parser.add_argument(
        "--sam2_ckpt",
        default="download_models/facebook/sam2.1/sam2.1_hiera_large.pt",
        help="SAM2 model checkpoint file path",
    )
    
    
    #
    parser.add_argument("--dataset_type", help= 'short, mid, long, full')
    parser.add_argument(
        "--vis_save_path", default=None, help="The dir to save results."
    )
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument("--grounding_with_max_confidence", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--box_threshold", default=0.35, type=float)
    parser.add_argument("--text_threshold", default=0.35, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # 1. build model
    groundedsam2_tracker = Tracker(
        grounding_model_config=args.gd_config,
        grounding_model_checkpoint=args.gd_ckpt,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # 2. save path
    args.vis_save_path = os.path.join(args.vis_save_path, args.dataset_type)
    os.makedirs(args.vis_save_path, exist_ok=True)
    
    # 3. load dataset
    dataset_info = dataset_mapping[args.dataset_type]
    
    meta_exp_path = dataset_info["meta_expression_file"]
    meta_exp = read_json(meta_exp_path)["videos"]
    
    raw_meta_path = dataset_info["meta_data_file"]
    raw_meta = read_json(raw_meta_path)["videos"]
    
    video_folder = dataset_info["source_frames_dir"]
    
    
    # 4. job list
    job_list = []
    for vid_id in meta_exp.keys():
        vid = meta_exp[vid_id]
        for exp_id in list(vid["expressions"].keys()):
            job_list.append((vid_id, exp_id))
    
    job_list_subset = [
        job_list[i]
        for i in range(len(job_list))
        if i % args.subset_num == args.subset_idx
    ]
    
    # 5. inference start
    total_infer = len(job_list_subset)
    progress_bar = tqdm(total=total_infer, desc="Progress {}".format(args.subset_idx))
    
    start_time = time.time()
    
    for vid_id, exp_id in job_list_subset:
        
        # dir for saving results
        save_dir_vid_exp = os.path.join(args.vis_save_path, vid_id, exp_id)
        
        # image save dir
        tmp_image_save_dir = os.path.join(save_dir_vid_exp, "source_frames")
        obj_id = meta_exp[vid_id]['expressions'][exp_id]['obj_id']
        
        # query
        ref_query = meta_exp[vid_id]['expressions'][exp_id]['exp']
        if ref_query[-1] == ".":
            ref_query = ref_query[:-1]
        ref_query = ref_query.lower()
        ref_query = ref_query.replace(". ", ";")
        
        # images
        if args.dataset_type == "full":
            vid_type = raw_meta[vid_id]["subset"]
        else:
            vid_type = args.dataset_type

        raw_clip_uid = vid_id.split("--")[0] if vid_type == "medium" else vid_id
        
        image_folder = os.path.join(video_folder, raw_clip_uid)
        
        if not os.path.exists(image_folder):
            print("File not found in {}".format(image_folder))
            raise FileNotFoundError
        
        image_list_wo_jpg = meta_exp[vid_id]["frames"]
        frame_names = [fn+".jpg" for fn in image_list_wo_jpg]
        
        new_grounding_ret = groundedsam2_tracker.tracking(
            image_dir=image_folder,
            frame_names=frame_names,
            save_source_frames_dir=tmp_image_save_dir,  # temp save
            tracking_results_dir=save_dir_vid_exp,  # save dir
            tracking_file_name=f"{exp_id}-{obj_id}.json",
            caption_name=ref_query.strip() + ".",
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            prompt_type_for_video="box",
            select_one_highest_conf=True,  
            grounding_with_max_confidence=args.grounding_with_max_confidence,
        )
        progress_bar.update(1)
    
    end_time = time.time()
    print("Inference Total time: ", end_time - start_time)