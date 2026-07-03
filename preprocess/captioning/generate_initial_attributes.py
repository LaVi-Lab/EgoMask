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
model_name = "gpt-4o-2024-11-20"
prompter = chatgpt_prompter(model_name=model_name, cache_filename=f"{model_name}.sqlit")

default_prompt_template = (
    "Please help me generate object descriptions.\n"
    "These are {total_frames} frames from a video. Each frame contains a red bounding box that correspondes to the same object. Based on the object in the red bounding box and its object tag, please generate its caption, visual attributes and affordance description (if applicable).\n"  # noqa: E501
    + "Object tag: {object_tag}\n"
    + "- Output should consist of three lines, separated by a newline:\n"
    + '1. A clear object caption with no more than 10 words, starting with "Object Caption: ".\n'  # noqa: E501
    + '2. The visual attributes of the object, starting with "Visual Attributes: ".\n'  # noqa: E501
    + '3. A concrete affordance description of the object, starting with "Object: Affordance: ".\n'  # noqa: E501
    + "**Restriction Policies**:\n"
    + "- Use the provided object tag selectively, as it may contain noise. "
    + "- The object caption should be a noun phrase.\n"
    + "- The object caption should clearly identify the object with the minimal words to avoid any ambiguity without referencing bounding boxes."
    + "- Visual attributes characterize the objects in images. They can be spatial location in the physical world, OCR characters on the object, spatial relations to surrounding objects, action relations to surrounding objects, relative size compared to surrounding objects, color, geometry shape, material, texture pattern, motion or dynamics of objects, and so on.\n"
    + "- The affordance description should focus on the object's potential actions, interactions, or functions, describing how the object can be utilized or manipulated in a given context. Avoid generic statements and provide specific and practical insights into the object's affordances.\n"
    + '- The affordance description should be a verb phrase, e.g., cut vegatables, clean the tables etc. If there is no affordance about the object, output "None". \n'
    + '- Do not use "red bounding box", "image", or "frame" in the answer.\n'
)


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
            force_rewrite=True,
        )

    # frame_names = get_all_frame_names(clip_source_frame_dir)

    # initialize (first appearance)
    first_appear = list(
        range(attribu_occ[0]["start_frame_number"], attribu_occ[0]["end_frame_number"]+1)
    )
    first_appear_duration = len(first_appear)
    print("First appear", first_appear)

    for idx, bbox in enumerate(bbox_list[:first_appear_duration]):
        if abs(bbox["frame_number"] - first_appear[idx]) > 5:
            print("Warning: frame number not match at 5", bbox["frame_number"], first_appear[idx])
            # import pdb;pdb.set_trace()
        # assert abs(bbox["frame_number"] - first_appear[idx]) <= 5

    first_appear_bbox = []
    for i, bbox in enumerate(bbox_list):
        if bbox["frame_number"] in first_appear:
            first_appear_bbox.append(bbox)
        if bbox["frame_number"] > first_appear[-1]:
            break

    annotated_frame_idx_list_with_space = [
        (idx, bbox["frame_number"], bbox["width"] / 200.0 * bbox["height"] / 200.0)
        for idx, bbox in enumerate(first_appear_bbox)
    ]
    annotated_frame_idx_list_with_space.sort(key=lambda x: x[-1], reverse=True)
    annotated_frame_idx_list = annotated_frame_idx_list_with_space[:max_num_frames]

    # # Overall appearance
    # append_idx_list = select_bigest_appearance(bbox_list, max_num_frames)
    # annotated_frame_idx_list.extend(append_idx_list)
    # annotated_frame_idx_list = list(set(annotated_frame_idx_list))

    # follow the video-stream
    annotated_frame_idx_list.sort(key=lambda x: x[0])

    print("select_annotated_idx", annotated_frame_idx_list)

    try:
        np_frames = [
            cv2.imread(os.path.join(clip_source_frame_dir, frame_names[frame_number]))
            for _, frame_number, _ in annotated_frame_idx_list
        ]
    except Exception as e:
        print("[ERROR] Load frames failed", e)
        import pdb;pdb.set_trace()

    for idx, (bbox_idx, _, _) in enumerate(annotated_frame_idx_list):
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
    if verbose:
        for i, frame in enumerate(np_frames):
            output_dir = f"./tmp/gpt4o_attribute_view/{clip_uid}/{object_title}"
            os.makedirs(output_dir, exist_ok=True)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), frame_bgr)

    base64_frames = build_base64_frames_from_np(np_frames)

    generate_parameter = {"max_tokens": 1024}

    query_text = prompt_template.format(
        total_frames=len(base64_frames), object_tag=object_title
    )
    print("[INFO] Query text:\n", query_text)
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
        default="ego4d_data/v2/clips/",
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

    parser.add_argument("--source_frame_dir", type=str, default="samples/source_frames")

    parser.add_argument("--data_dir", type=str, default="samples", help="load bounding box information")
    
    parser.add_argument("--bbox_info_file", type=str, default="samples/egotracks_clip_object_bbox.json")
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="samples",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to save the frames for prompting."
    )
    parser.add_argument("--max_num_frames", type=int, default=3)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import json
    import time
    args = parse_args()

    """
    [完成]
    python captioning/generate_initial_attributes.py \
    --data_dir sample \
    --save_dir sample \
    --source_frame_dir sample/source_frames \
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

    save_file = os.path.join(args.save_dir,"initial_attributions.json")
    with open(save_file,"w") as f:
        json.dump(saved_response,f)
    print("[INFO] Save to",save_file)
    print("[INFO] Time cost",time.time()-start_)
