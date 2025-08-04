import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import pycocotools.mask as maskUtils
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from evaluation.common_utils import *

VIDEO_LISA_ROOT = os.environ.get("VIDEO_LISA_ROOT", "../..")
sys.path.append(VIDEO_LISA_ROOT)
from model.VideoLISA import VideoLISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import codecs
from typing import Any, List
import json
import time

        
def parse_args():
    parser = argparse.ArgumentParser(description="VideoLISA Inference")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument("--dataset_type", default="long")
    parser.add_argument("--vis_save_path", default="./results/VideoLISA-egomask/", type=str)
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--subset_num", default=8, type=int)
    parser.add_argument("--subset_idx", default=0, type=int)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--num_frames_sparse", default=50, type=int)
    parser.add_argument("--num_frames_dense", default=4, type=int)
    parser.add_argument(
        "--conv_type",
        default="phi3_instruct",
        type=str,
    )

    args = parser.parse_args()
    return args


def simple_process(ref_query):
    ref_query = ref_query.strip()
    if ref_query[-1] == ".":
        ref_query = ref_query[:-1]
    return ref_query.lower()


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



# uniformly sample frames
def get_sparse_indices(total_frame_num, num_frames_sparse):
    
    def uniform_sample(total_len, sample_num):
        intervals = np.linspace(start=0, stop=total_len, num=sample_num + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

        return frame_idxs

    if total_frame_num > num_frames_sparse:       # video is long, uniformly sample frames
        frame_idxs = uniform_sample(total_frame_num, num_frames_sparse)
        return sorted(frame_idxs)
    else:
        num_repeat = num_frames_sparse // total_frame_num
        num_sample = num_frames_sparse % total_frame_num
        frame_idxs = list(range(total_frame_num)) * num_repeat + uniform_sample(total_frame_num, num_sample)
        return sorted(frame_idxs)


def get_dense_indices(num_frames_sparse, num_frames_dense):
    intervals = np.linspace(start=0, stop=num_frames_sparse - 1, num=num_frames_dense + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]

    return frame_idxs

if __name__ == "__main__":
    args = parse_args()
    
    
    # 1. Build model
    ## tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
        
    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    ## model
    model = VideoLISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        empty_init=False,
        **kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    else:
        raise NotImplementedError

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    
    
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
        total_frames = len(image_file_list)
        
        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_sparse)
        valid_dense_idxs = get_dense_indices(args.num_frames_sparse, args.num_frames_dense)
        
        # prepare text query and prompt
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        obj_id = meta_exp[vid_id]['expressions'][exp_id]['obj_id']
        ref_query = meta_exp[vid_id]['expressions'][exp_id]['exp']
        
        prompt_template = "Please segment the {class_name} in this image."
        prompt = prompt_template.format(class_name=simple_process(ref_query))
        print("[current prompt]", prompt)
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "Sure, [SEG].")
        prompt = conv.get_prompt()
        
        # pre-process images
        image_list_sam, image_list_clip, image_list_np = [], [], []

        for frm_idx in sparse_idxs:
            image_path = image_file_list[frm_idx]
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image_list_clip.append(image_clip)
        
        for frm_idx in range(total_frames):
            image_path = image_file_list[frm_idx]
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            image_list_sam.append(image)
            image_list_np.append(image_np)
        
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # stack as video
        image = torch.stack(image_list_sam, dim=1)
        image_clip = torch.stack(image_list_clip, dim=1)

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image, # sam2 tracking
            input_ids,
            resize_list,
            original_size_list,
            dense_indices=[valid_dense_idxs],
            num_frames_sparse=args.num_frames_sparse,
            num_frames_dense=args.num_frames_dense,
        )
        for i, pred_mask_vid in enumerate(pred_masks):
            assert i == 0
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]
            new_tracking = {}
            for frame_idx in tqdm(range(total_frames)):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0
                
                if pred_mask.sum() > 0:
                    mask_segm = maskUtils.encode(np.asfortranarray(pred_mask))
                    mask_segm["counts"] = mask_segm["counts"].decode()

                    save_k = image_list_wo_jpg[frame_idx]
                    new_tracking[save_k] = mask_segm
            print("[INFO] saving tracking results for video {} expression {}".format(vid_id, exp_id))
            write_json(new_tracking, f"{save_dir_vid_exp}/{exp_id}-{obj_id}.json")
            print("[INFO] tracking results saved in {}".format(save_dir_vid_exp))

            if args.save_overlay:
                for frame_idx in tqdm(range(total_frames)):

                    save_k = image_list_wo_jpg[frame_idx].replace("jpg","")

                    save_path = "{}/masked_img_{}.jpg".format(save_dir_vid_exp, save_k)
                    save_img = image_list_np[frame_idx].copy()
                    save_img[pred_mask] = (
                            image_list_np[frame_idx] * 0.5
                            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask]
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save_img)
        torch.cuda.empty_cache()
        progress_bar.update(1)
    end_time = time.time()
    print("Inference Total time: ", end_time - start_time)