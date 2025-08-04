import os
import sys
sys.path.append(os.getcwd())
import time
import cv2
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pycocotools.mask as maskUtils
import numpy as np
import torch
from evaluation.common_utils import *

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")
    
def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    
    parser.add_argument("--dataset_type", help= 'short, mid, long, full')
    parser.add_argument(
        "--vis_save_path", default=None, help="The dir to save results."
    )
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-4B")
    
    args = parser.parse_args()
    return args

def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='r', alphas=0.7)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

def simple_process(ref_query):
    ref_query = ref_query.strip()
    if ref_query[-1] == ".":
        ref_query = ref_query[:-1]
    return ref_query.lower()

if __name__ == "__main__":
    args = parse_args()
    
    # 1. build model
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
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
    total_infer = len(job_list_subset)
    progress_bar = tqdm(total=total_infer, desc="Progress {}".format(args.subset_idx))
    
    start_time = time.time()
    for vid_id, exp_id in job_list_subset:
        
        # dir for saving results
        save_dir_vid_exp = os.path.join(args.vis_save_path, vid_id, exp_id)
        os.makedirs(save_dir_vid_exp, exist_ok=True)
        
        obj_id = meta_exp[vid_id]['expressions'][exp_id]['obj_id']
        ref_query = meta_exp[vid_id]['expressions'][exp_id]['exp']
        prompt_template = "<image> Please segment the {class_name} in this image."
        prompt = prompt_template.format(class_name=simple_process(ref_query))
        print("[current prompt]", prompt)
        
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

        image_file_list = [
            os.path.join(image_folder, fn + ".jpg") for fn in image_list_wo_jpg
        ]

        # read images
        vid_frames = []
        for img_path in image_file_list:
            img = Image.open(img_path).convert("RGB")
            vid_frames.append(img)
        print("[INFO] len(vid_frames)", len(vid_frames),len(image_list_wo_jpg))
        
        result = model.predict_forward(
            video=vid_frames,
            text=prompt,
            tokenizer=tokenizer,
        )
        prediction = result["prediction"]
        
        # save tracking results
        new_tracking = {}
        if "[SEG]" in prediction:
            _seg_idx = 0
            pred_masks = result["prediction_masks"][_seg_idx]
            
            for frame_idx in range(len(vid_frames)):
                pred_mask = pred_masks[frame_idx]
                if pred_mask.sum() > 0:
                    mask_segm = maskUtils.encode(np.asfortranarray(pred_mask))
                    mask_segm["counts"] = mask_segm["counts"].decode()

                    save_k = image_list_wo_jpg[frame_idx]
                    new_tracking[save_k] = mask_segm
                if Visualizer is not None and args.vis:
                    os.makedirs("./temp_visualize_results", exist_ok=True)
                    visualize(
                        pred_mask,
                        image_file_list[frame_idx],
                        save_dir_vid_exp,
                    )
                    
        print("[INFO] saving tracking results for video {} expression {}".format(vid_id, exp_id))

        tmp_fname = f"{save_dir_vid_exp}/{exp_id}-{obj_id}.json"
        write_json(new_tracking, tmp_fname)
        print("[INFO] tracking results saved in {}".format(tmp_fname))

        torch.cuda.empty_cache()
        progress_bar.update(1)
    
    end_time = time.time()
    print("Inference Total time: ", end_time - start_time)