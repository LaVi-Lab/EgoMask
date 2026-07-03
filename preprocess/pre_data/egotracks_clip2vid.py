"""
Build clip_uid(*.mp4) to video_uid
"""
import os, sys
sys.path.append(os.getcwd())
from utils.dataset_io import build_consistent_appearance, refine_bbox_into_5FPS, refine_attributes_occurrence
from utils.io import read_json, read_txt, write_txt

import json
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
import random
import spacy

nlp = spacy.load("en_core_web_sm")


def ego4d_info(
    ego4d_file: str = r"ego4d_data/ego4d.json",
    save_dir: str = r"preprocess/pre_data",
    max_num: int = 0,
):
    with open(ego4d_file, "r") as f:
        ego4d_clip_video = json.load(f)

    videos_scenarios = {
        v["video_uid"]: v["scenarios"] for v in ego4d_clip_video["videos"]
    }

    clips_info = ego4d_clip_video["clips"]
    clip2video_uid = {}
    if max_num > 0:
        clips_info = clips_info[:max_num]

    clip_no_scenarios_num = 0

    for d in clips_info:
        try:
            video_uid = d["video_uid"]
            clip_uid = d["clip_uid"]
            assert clip_uid not in clip2video_uid
            clip2video_uid[clip_uid] = {
                "video_uid": video_uid,
                "video_start_frame": d["video_start_frame"],
                "video_end_frame": d["video_end_frame"],
                "clip_fps": d["clip_metadata"]["fps"],
                "videos_scenarios": (
                    videos_scenarios[video_uid] if video_uid in videos_scenarios else None
                ),
                "duration": (d["video_end_frame"] - d["video_start_frame"])
                / d["clip_metadata"]["fps"],
            }
            if video_uid not in videos_scenarios:
                clip_no_scenarios_num += 1
        except Exception as e:
            print(e, d)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "ego4d_clip2video_uid.json"), "w") as f:
        json.dump(clip2video_uid, f)

    print("[INFO] (Raw_data) Ego4d: Total Clips", len(clip2video_uid))  # 12291
    print("[INFO] (Raw data) Ego4d: clip_no_scenarios", clip_no_scenarios_num)  # 11

    return clip2video_uid


def clip2video(
    egotracks_dir: str = r"ego4d_data/v2/egotracks",
    save_dir: str = r"preprocess/pre_data",
    max_num: int = 0,
):
    train_path = os.path.join(egotracks_dir, "egotracks_train.json")
    eval_path = os.path.join(egotracks_dir, "egotracks_val.json")
    test_path = os.path.join(egotracks_dir, "egotracks_challenge_test_unannotated.json")

    data_paths = [train_path, eval_path, test_path]
    data_names = ["train", "eval", "test"]

    egotracks_clip2video_uid = {}
    egotracks_clip_object_bbox = {}

    egotracks_stat = {}
    
    all_not_found = 0
    for data_key, filepath in zip(data_names, data_paths):

        egotracks_stat[data_key] = {}

        with open(filepath, "r") as f:
            current_data = json.load(f)

        video_num = 0
        clip_num = 0
        object_stat_per_clip = []
        object_num = 0
        lt_tracking_num = 0
        no_tracking_clip_num = 0
        avg_duration = 0

        test_duration = 0

        for vidx, video in enumerate(tqdm(current_data["videos"])):
        # for vidx, video in enumerate(current_data["videos"][:1]):
            if max_num > 0 and vidx >= max_num:
                break
            video_num += 1
            video_uid = video["video_uid"]

            for cidx, clip in enumerate(video["clips"]):
                clip_num += 1

                clip_uid = clip["clip_uid"]
                assert clip_uid not in egotracks_clip2video_uid
                tmp_dict = {
                    "video_uid": video_uid,
                    "type": data_key,
                    "video_start_frame": clip["video_start_frame"],
                    "video_end_frame": clip["video_end_frame"],
                    "video_idx": vidx,
                    "clip_uidx": cidx,
                }
                test_duration += (
                    clip["video_end_frame"] - clip["video_start_frame"]
                ) / 30
                egotracks_clip2video_uid[clip_uid] = tmp_dict

                object_num_one_clip = 0
                if data_key == "test":
                    if "annotations" in clip:
                        for annots_id, annotation in enumerate(clip["annotations"]):
                            if "query_sets" in annotation:
                                annot = annotation["query_sets"]
                                for anno_id, value in annot.items():
                                    # if value["is_valid"]:
                                        object_num_one_clip += 1
                                        object_num += 1
                
                
                if data_key != "test" and "annotations" in clip:
                    tmp_object_bboxs = {}
                    for annots_id, annotation in enumerate(clip["annotations"]):
                        # find object title
                        if "query_sets" in annotation:
                            annot = annotation["query_sets"]
                            for anno_id, value in annot.items():
                                print("[INFO]anno_id", vidx, cidx, annots_id, anno_id)
                                # if "object_title" in value:
                                if value["is_valid"]:
                                    object_num_one_clip += 1
                                    object_title = value["object_title"]
                                    object_num += 1
                                    if "lt_track" in value:
                                        lt_tracking_num += 1
                                        bboxes = value["lt_track"]
                                        # ! FIXED：original annotations are not sorted
                                        bboxes.sort(key=lambda x: x["frame_number"])
                                        # ! REFINE BBOX (approx 5FPS) & remove duplicate
                                        bboxes = refine_bbox_into_5FPS(bboxes, video_stride=6)

                                        tmp_object_bboxs[object_title] = {
                                            "bbox_anno": bboxes
                                        }
                                        if "attributes_occurrence" not in value:
                                            tmp_object_bboxs[object_title][
                                                "attributes_occurrence"
                                            ] = build_consistent_appearance(bboxes)

                                        else:
                                            occur_anno = value["attributes_occurrence"]
                                            occur_anno.sort(
                                                key=lambda x: x["start_frame_number"]
                                            )
                                            # ! FIXED: approx 5FPS
                                            new_occur_anno, not_added = refine_attributes_occurrence(bboxes, occur_anno)
                                            tmp_object_bboxs[object_title][
                                                "attributes_occurrence"
                                            ] = new_occur_anno
                                            print("[NOT_ADDED]", not_added)
                                            all_not_found += not_added

                    if len(tmp_object_bboxs) != 0:
                        egotracks_clip_object_bbox[clip_uid] = tmp_object_bboxs
                        avg_duration += (
                            clip["video_end_frame"] - clip["video_start_frame"]
                        ) / 30
                    else:
                        no_tracking_clip_num += 1
                        print(
                            f"[WARNING] No lt_tracking data for {no_tracking_clip_num} clips"
                        )

                object_stat_per_clip.append(object_num_one_clip)
                print("Finish One!")
        egotracks_stat[data_key] = {
            "video_num": video_num,
            "clip_num": clip_num,
            "object_num_per_clip": (
                max(object_stat_per_clip),
                min(object_stat_per_clip),
                sum(object_stat_per_clip) / clip_num,
            ),
            "tracking_per_clip": lt_tracking_num / (clip_num - no_tracking_clip_num),
            "object_num": object_num,
            "lt_tracking_num": lt_tracking_num,
            "no_tracking_clip_num": no_tracking_clip_num,
            "avg_duration": avg_duration / (clip_num - no_tracking_clip_num),
            "test_duration": test_duration / clip_num,
        }

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "egotracks_clip2video_uid.json"), "w") as f:
        json.dump(egotracks_clip2video_uid, f)

    with open(os.path.join(save_dir, "egotracks_clip_object_bbox.json"), "w") as f:
        json.dump(egotracks_clip_object_bbox, f)

    print("[INFO] (Raw_data) EgoTracks: clip2video_uid & clip_object_bbox file [Save!]")

    with open(os.path.join(save_dir, "egotracks_stat.json"), "w") as f:
        json.dump(egotracks_stat, f)

    print("[INFO] (Raw_data) EgoTracks: \n", egotracks_stat)
    print("[INFO] (Raw_data) EgoTracks: all_not_found (When Refining occur_anno)", all_not_found)




if __name__ == "__main__":
    # raw_data
    ego4d_info()
    clip2video()
