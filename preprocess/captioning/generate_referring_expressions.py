import os, sys
import json

sys.path.append(os.getcwd())
from utils.dataset_io import build_consistent_appearance
from utils.image_process_func import (
    draw_box,
    egotracks_xywh2xyxy,
    build_base64_frames_from_np,
)
from utils.video_process_func import get_all_frame_names, preprocess_video
from utils.io import read_txt
from data_prepare.captioning.chatgpt_client import chatgpt_prompter
import cv2
from typing import List,Optional,Dict
import argparse
import shutil
import json
import time
model_name = "gpt-4o-2024-11-20"
prompter = chatgpt_prompter(model_name=model_name, cache_filename=f"{model_name}.sqlit")

default_prompt_template = (
    "Please help me generate referring expressions for object segmentation.\n"
    "These are {total_frames} frames from a video. Each frame contains a red bounding box that correspondes to the same object. Based on the object in the red bounding box and its object tag, please generate the descriptions that uniquely identify the object throughout the video.\n"  # noqa: E501
    + "Object tag: {object_tag}\n"
    + "- Output should consist of two lines, separated by a newline:\n"
    + '1. A short experessions with no more than 10 words, starting with "Short expressions: ".\n'  # noqa: E501
    + '2. A longer expressions with more detailed illustrations, starting with "Long expressions: ".\n'  # noqa: E501
    + "**Restriction Policies**:\n"
    + "- The referring expressions should be concise and informative. They can be spatial location in the physical world, OCR characters on the object, spatial relations to surrounding objects, action relations to surrounding objects, relative size compared to surrounding objects, color, geometry shape, material, texture pattern, motion or dynamics of objects, and so on.\n"
    + "- The generated referring expressions should clearly identify the object to avoid any ambiguity without referencing bounding boxes in the video. \n"
    + '- Do not use "red bounding box", "image", or "frame" in the answer.\n'
)

def chunk_list(my_list, chunk_num=4):
    chunk_size = len(my_list) // chunk_num
    chunks = [my_list[i:i + chunk_size] for i in range(0, len(my_list), chunk_size)]
    return chunks

def select_bigest_appearance(attribu_occ, bbox_list, max_num_frames):
    bbox_list_map = {
        x["frame_number"]: (bbox_idx, x["width"] * x["height"])
        for bbox_idx, x in enumerate(bbox_list)
    }
    selected_chunk_num = min(len(attribu_occ), max_num_frames)
    chunked_list = chunk_list(attribu_occ, chunk_num=selected_chunk_num)
    # 得到每一段对应的开始和结束坐标
    select_log = []
    selected_frame_number = []
    for chunk_idx, chunk_sub_list in enumerate(chunked_list):
        start_ = chunk_sub_list[0]["start_frame_number"]
        end_ = chunk_sub_list[-1]["end_frame_number"]
        # chunk_log.append((chunk_idx, start_, end_))

        current_bbox_list = [
            (k, v[0], v[1]) for k, v in bbox_list_map.items() if start_ <= k <= end_
        ]
        current_bbox_list.sort(key=lambda x: x[-1], reverse=True)
        # print(current_bbox_list[:3])
        select_log.append(current_bbox_list[0][:2])
        selected_frame_number.append(current_bbox_list[0][0])

    if len(select_log) < max_num_frames:
        bbox_list_sort = [
            (k, v[0], v[1])
            for k, v in bbox_list_map.items()
            if k not in selected_frame_number
        ]
        current_bbox_list.sort(key=lambda x: x[-1], reverse=True)

        remaining_ = max_num_frames - len(select_log)
        for frame_number, bbox_idx, _ in current_bbox_list[:remaining_]:
            select_log.append((frame_number, bbox_idx))

    select_log.sort(key=lambda x: x[0])
    return select_log


def process_one(
    source_video_dir:str,
    source_frame_dir:str,
    clip_uid:str,
    object_title:str,
    object_bbox_info = None,
    max_num_frames: int = 3,
    prompt_template: str= "",
    verbose: bool = False,
    video_stride: int=6,
):
    # import pdb;pdb.set_trace()
    if type(object_bbox_info) is dict:
        bbox_list = object_bbox_info["bbox_anno"]
        if "attributes_occurrence" not in object_bbox_info:
            attribu_occ = object_bbox_info["attributes_occurrence"]
        else:
            attribu_occ = build_consistent_appearance(bbox_list)
    else:
        bbox_list = object_bbox_info
        attribu_occ = build_consistent_appearance(bbox_list)

    clip_source_frame_dir = os.path.join(source_frame_dir, clip_uid)
    video_path = os.path.join(source_video_dir, clip_uid + ".mp4")
    frame_names = preprocess_video(
            video_path=video_path,
            save_source_frames_dir=clip_source_frame_dir,
            video_stride=video_stride,
            force_rewrite=False,
        )

    # # Overall appearance
    annotated_frame_idx_list = select_bigest_appearance(attribu_occ, bbox_list, max_num_frames)
    # follow the video-stream
    annotated_frame_idx_list.sort(key=lambda x: x[1])

    print("select_annotated_idx", annotated_frame_idx_list)
    skip_flag = False
    try:
        np_frames = [
            cv2.imread(os.path.join(clip_source_frame_dir, frame_names[frame_number]))
            for frame_number, bbox_idx in annotated_frame_idx_list
        ]
    except Exception as e:
        skip_flag = True
        print("[ERROR] Load frames failed", e)
        # import pdb;pdb.set_trace()

    if skip_flag:
        print(">>>>>>>>>[WARNINGING]", clip_uid, object_title)
        return None
        
    for idx, (_, bbox_idx) in enumerate(annotated_frame_idx_list):
        bbox_anno = bbox_list[bbox_idx]
        bbox_xywh = [bbox_anno[k] for k in ["x", "y", "width", "height"]]
        np_frames[idx] = draw_box(
            np_frames[idx],
            boxes=[egotracks_xywh2xyxy(bbox_xywh)],
            color="red",
            label=[object_title],
            RGB_in=False,
            RGB_out=True,
        )
    if False:
        for i, frame in enumerate(np_frames):
            output_dir = f"tmp/gpt4o_attribute_view/{clip_uid}/{object_title}"
            os.makedirs(output_dir, exist_ok=True)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), frame_bgr)

    base64_frames = build_base64_frames_from_np(np_frames)

    generate_parameter = {"max_tokens": 1024}

    query_text = prompt_template.format(
        total_frames=len(base64_frames), object_tag=object_title
    )
    print("[INFO] Query text:\n", query_text)
    # import pdb;pdb.set_trace()
    try:
        response, cached, if_call_api = prompter.prompt(
                query_text=query_text,
                base64_frames=base64_frames,
                **generate_parameter,
            )
        if not if_call_api:
            print("[HIT Cache]")
        print("[Response]\n",response)
        return response

    except Exception as e:
        print("[ERROR]", e)
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_stride", type=int, default=6)
    parser.add_argument(
        "--source_video_dir",
        type=str,
        default="/ego4d_data/v2/clips/",
    )
    parser.add_argument(
        "--clip_uid", type=str, default=None
    )
    parser.add_argument("--object_caption", type=str, default=None)
    parser.add_argument("--bbox_json_file", type=str, default=None)
    parser.add_argument(
        "--clip_uid_file",
        type=str,
        default=None,
    )
    parser.add_argument("--video_path", type=str, default="")

    parser.add_argument("--source_frame_dir", type=str, default="sample/source_frames") # extracted from ego4d videos

    parser.add_argument("--data_dir", type=str, default="sample/", help="load bounding box information")
    
    parser.add_argument("--bbox_info_file", type=str, default="sample/egotracks_clip_object_bbox.json")
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="samples/sample_100",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to save the frames for prompting."
    )
    parser.add_argument("--max_num_frames", type=int, default=3)


    args = parser.parse_args()
    return args

def run_main():
    args = parse_args()
    """
    [DONE]
    python captioning/generate_referring_expressions.py \
    --data_dir sample \
    --save_dir sample \
    --source_frame_dir ego4d_clips/source_frames \
    --verbose
    """

    saved_response = {}
    if args.clip_uid_file is not None:
        clip_data = read_txt(args.clip_uid_file)
    elif args.clip_uid is None:
        print("[INFO] Use selected_clip_uids.txt")
        clip_data = read_txt(os.path.join(args.data_dir, "selected_clip_uids.txt"))
    else:
        assert args.clip_uid is not None
        clip_data = [args.clip_uid]

    # further to support more format
    if args.bbox_info_file is not None:
        print("[INFO] Load bbox info from",args.bbox_info_file)
        bbox_info_file = args.bbox_info_file
    else:
        bbox_info_file = os.path.join(args.data_dir, "selected_egotracks_bboxs.json")
    with open(bbox_info_file,"r") as f:
        bbox_info_data = json.load(f)

    # import pdb;pdb.set_trace()
    start_ = time.time()
    for clip_uid in clip_data:
        if clip_uid in bbox_info_data:

            box_info = bbox_info_data[clip_uid]

            print("[INFO] clip_uid",clip_uid)
            saved_response[clip_uid] = {}

            for object_title, object_bbox_info in box_info.items():

                response = process_one(
                    source_video_dir=args.source_video_dir,
                    source_frame_dir=args.source_frame_dir,
                    clip_uid=clip_uid,
                    object_title=object_title,
                    object_bbox_info=object_bbox_info,
                    max_num_frames=args.max_num_frames,
                    prompt_template=default_prompt_template,
                    verbose=args.verbose,
                    video_stride=args.video_stride,
                )
                # print(response)
                saved_response[clip_uid][object_title] = response

            clip_source_frame_dir = os.path.join(args.source_frame_dir, clip_uid)
            if os.path.exists(clip_source_frame_dir):
                shutil.rmtree(clip_source_frame_dir)

    save_file = os.path.join(args.save_dir,"overall_expressions.json")
    with open(save_file,"w") as f:
        json.dump(saved_response,f)
    print("[INFO] Save to",save_file)
    print("[INFO] Time cost",time.time()-start_)

if __name__ == "__main__":
    run_main()
