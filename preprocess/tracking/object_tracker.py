import os,sys
sys.path.append(os.getcwd())

from utils.const import GROUNDED_SAM_2_DIR, GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT,SAM2_1_CHECKPOINT,SAM2_1_CONFIG_DIR
from utils.image_process_func import mask2rle,egotracks_xywh2xyxy,mask2bbox,convert_xyxy2xywh
from utils.video_process_func import get_all_frame_names, preprocess_video, create_video_from_images,mask_other_region
from utils.dataset_io import build_consistent_appearance
from utils.io import read_txt, read_json
sys.path.append(GROUNDED_SAM_2_DIR)


from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks

import cv2
import torch
import json
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from typing import Optional, Dict, List
import shutil
import argparse
import pycocotools.mask as maskUtils

# Grounding model [In case no initial bbox provided]

class GroundingDino_model:
    def __init__(self, grounding_model_config: str,  
                grounding_model_checkpoint: str, 
                device: str = "cuda"):

        self.grounding_model_config = grounding_model_config
        self.grounding_model_checkpoint = grounding_model_checkpoint
        self.device = device

        self.grounding_model = load_model(
            model_config_path=self.grounding_model_config,
            model_checkpoint_path=self.grounding_model_checkpoint,
            device=self.device,
        )

    def prompt_grounding_model(
        self, save_dir, frame_names, 
        caption, box_threshold, text_threshold, 
        grounding_with_max_confidence:bool=False
    ):
        '''
        save_dir: str, where to put the frames from the video
        frame_names: List[str], frame filepath
        caption: str
        box_threshold: float
        text_threshold: float
        '''
        kept_result = None
        max_confidence = 0.0
        for idx, frame_name in enumerate(frame_names):

            img_path = os.path.join(save_dir, frame_name)
            with torch.cuda.amp.autocast(enabled=False):
                image_source, image = load_image(img_path)

                boxes, confidences, labels = predict(
                    model=self.grounding_model,
                    image=image,
                    caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            if len(labels) > 0:
                print("[INFO] GroundingDino find object at Frame:", idx)
                ret = {
                    "image_source": image_source,
                    "boxes": boxes,
                    "confidences": confidences,
                    "labels": labels,
                    "frame_idx": idx,
                }
                if max(confidences) > max_confidence:
                    kept_result = ret
                    max_confidence = max(confidences)

                if not grounding_with_max_confidence:
                    print("[INFO] GroundingDino fisrt find object at Frame:", idx)
                    return ret

        if kept_result:
            idx = kept_result["frame_idx"]
            print("[INFO] GroundingDino find object with max_confidence at Frame:", idx)
            return kept_result

        return None

    def convert_return_format(self, grounding_ret, select_one=True):
        image_source = grounding_ret["image_source"]
        ann_frame_idx = grounding_ret["frame_idx"]
        height, width, _ = image_source.shape
        boxes = grounding_ret["boxes"] * torch.Tensor(
            [width, height, width, height]
        )
        input_boxes = box_convert(
            boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
        ).numpy()
        objects = grounding_ret["labels"]

        if select_one and len(objects) > 1:
            confidences = grounding_ret["confidences"].numpy().tolist()
            confidences_with_idx = [(idx,conf) for idx, conf in enumerate(confidences)]
            confidences_with_idx.sort(key=lambda x: -x[-1])
            max_conf_idx = confidences_with_idx[0][0]
            input_boxes = [input_boxes[max_conf_idx]]
            objects = [objects[max_conf_idx]]

        ret_value = {
            "image_source": image_source,
            "frame_idx": ann_frame_idx,
            "input_boxes": input_boxes,
            "labels": objects,
        }
        return ret_value


def convert_initial_bbox2grounding(initial_bbox_info:dict, 
                                caption:str,
                                img_path:str):
    ann_frame_idx = initial_bbox_info["frame_number"]
    print("[INFO] Initial BBox at ", ann_frame_idx)

    image_source, _ = load_image(img_path)

    # height, width, _ = image_source.shape

    x, y, w, h = [initial_bbox_info[k] for k in ["x", "y", "width", "height"]]
    xyxy = egotracks_xywh2xyxy([x, y, w, h])
    input_boxes = np.array([xyxy])

    objects = [caption]

    grounding_ret = {
        "frame_idx": ann_frame_idx,
        "image_source": image_source,
        "boxes": [x, y, w, h],
        "objects": objects,
        "input_boxes": input_boxes,
    }
    return grounding_ret

class SAM2_Tracker:
    def __init__(self, 
                sam2_config: str, 
                sam2_checkpoint: str,
                device: str = "cuda"):

        self.device = device
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.video_predictor = build_sam2_video_predictor(
            self.sam2_config, self.sam2_checkpoint
        )

        sam2_image_model = build_sam2(self.sam2_config, self.sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # what is image source ? what is input_boxes
    def image_generate_mask_from_boxes(self, image_source, input_boxes):

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(image_source)
        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()

        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return {"masks": masks, "scores": scores, "logits": logits}

    def video_predictor_registration(
        self,
        prompt_type_for_video,
        objects,
        input_boxes,
        masks,
        inference_state,
        ann_frame_idx: int,
    ):
        assert prompt_type_for_video in [
            "point",
            "box",
            "mask",
        ], "SAM 2 video predictor only support point/box/mask prompt"
        if prompt_type_for_video == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

            for object_id, (label, points) in enumerate(
                zip(objects, all_sample_points), start=1
            ):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = (
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )
                )
        # Using box prompt
        elif prompt_type_for_video == "box":
            for object_id, (label, box) in enumerate(
                zip(objects, input_boxes), start=1
            ):
                _, out_obj_ids, out_mask_logits = (
                    self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=object_id,
                        box=box,
                    )
                )
        # Using mask prompt is a more straightforward way
        elif prompt_type_for_video == "mask":
            for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask,
                )
        else:
            raise NotImplementedError(
                "SAM 2 video predictor only support point/box/mask prompts"
            )

    def tracking(
        self,
        save_source_frames_segments_dir: str,
        grounding_ret: Dict = None,
        prompt_type_for_video: str = "box",
        all_bbox_info: Optional[List] = None,
        zero_ann_frame_idx: bool = True,
        under_bbox_region: bool = False,
        mask_other_regions: bool = False
    ):

        # process the box prompt for SAM 2
        image_source = grounding_ret["image_source"]
        height, width, _ = image_source.shape
        input_boxes = grounding_ret["input_boxes"]
        objects = grounding_ret["labels"]

        mask_ret = self.image_generate_mask_from_boxes(image_source, input_boxes)

        inference_state = self.video_predictor.init_state(
            video_path=save_source_frames_segments_dir
        )

        self.video_predictor_registration(
            prompt_type_for_video=prompt_type_for_video,
            objects=objects,
            input_boxes=input_boxes,
            masks=mask_ret["masks"],
            inference_state=inference_state,
            ann_frame_idx=0 if zero_ann_frame_idx else grounding_ret["frame_idx"],
        )
        # video_segments contains the per-frame segmentation results
        video_segments = {}

        # obtain supervised bbox
        out_frame_bbox_info = {}
        if all_bbox_info is not None:
            out_frame_bbox_info = {

                int(bbox["segment_frame_idx"]): egotracks_xywh2xyxy(
                    [bbox[k] for k in ["x", "y", "width", "height"]]
                )
                for bbox in all_bbox_info
            }
        print("[INFO] (Bbox supervision) out_frame_bbox_info", out_frame_bbox_info)

        kept_logits = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state):

            video_segments[out_frame_idx] = {}
            kept_logits[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):

                current_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                if out_frame_idx in out_frame_bbox_info and under_bbox_region:
                    # post-process: make the mask only focus on the bbox region (for better visual quality)
                    x_min, y_min, x_max, y_max = out_frame_bbox_info[out_frame_idx]
                    append_mask = np.zeros_like(current_mask)
                    x_min, y_min, x_max, y_max = (
                        round(x_min),
                        round(y_min),
                        round(x_max),
                        round(y_max),
                    )
                    append_mask[:, y_min:y_max, x_min:x_max] = 1
                    current_mask = current_mask * append_mask

                video_segments[out_frame_idx][out_obj_id] = current_mask
                kept_logits[out_frame_idx][out_obj_id] = out_mask_logits[i]

        tracking_results = {
            "grounding": grounding_ret,
            "initial_mask": mask_ret,
            "tracking": video_segments,
            "height": height,
            "width": width,
            "tracking_logits": kept_logits,
        }
        self.video_predictor.reset_state(inference_state)
        return tracking_results


class Tracker:
    def __init__(self, grounding_model_config: str,  
                grounding_model_checkpoint: str,
                sam2_config: str, 
                sam2_checkpoint: str,
                device: str = "cuda"):
        self.grounding_model = GroundingDino_model(grounding_model_config,
                                                grounding_model_checkpoint,
                                                device)

        self.sam2_tracker = SAM2_Tracker(sam2_config, sam2_checkpoint)

    # 分段tracking
    def tracking_video_with_occurrence(
        self,
        video_path: str,
        save_source_frames_dir: str,
        caption_name: str,
        video_stride: int = 6,
        start: int = 0,
        end: int = None,
        prompt_type_for_video: str = "box",
        initial_bbox_info: Optional[List] = None,
        bbox_info: Optional[List] = None,
        under_bbox_region: bool = True,
        mask_other_regions: bool = False,
        **kwargs,
    ):
        if bbox_info is not None and mask_other_regions:
            init_state_video_dir = os.path.join(save_source_frames_dir, "tmp")
            mask_other_region(
                source_frame_dir=save_source_frames_dir,
                tmp_source_frame_dir=init_state_video_dir,
                all_bbox_info=bbox_info
            )
            print("[INFO] Mask Other Regions")
        else:
            init_state_video_dir = save_source_frames_dir

        # [{start_idx, end_idx}, ...] [)
        frame_names = preprocess_video(
            video_path=video_path,
            save_source_frames_dir=save_source_frames_dir,
            video_stride=video_stride,
            start=start,
            end=end,
            force_rewrite=False
        )
        # ! filter
        bbox_info = [d for d in bbox_info if d["frame_number"] < len(frame_names)]

        segments_info = build_consistent_appearance(bbox_info)
        # frame_number --> idx in video
        # frame_number(idx in video) --> idx in bbox_info
        frame2idx = {frame_name:idx for idx, frame_name in enumerate(frame_names)} 
        bbox_idx2frame = {bbox_info["frame_number"]:idx for idx, bbox_info in enumerate(bbox_info)}

        frame_name_mapping = {}
        tracking_combination = {}
        tracking_logits = {}

        # first image
        img_path = os.path.join(save_source_frames_dir, frame_names[0])
        image_source, _ = load_image(img_path)
        height, width, _ = image_source.shape

        for segment_idx, segment_info in enumerate(segments_info):
            # []
            start_idx = segment_info["start_frame_number"]
            end_idx = segment_info["end_frame_number"] 

            img_path = os.path.join(save_source_frames_dir, frame_names[start_idx])
            image_source, _ = load_image(img_path)

            initial_bbox_info = bbox_info[bbox_idx2frame[start_idx]] # start_idx = the index of 

            x, y, w, h = [initial_bbox_info[k] for k in ["x", "y", "width", "height"]]
            xyxy = egotracks_xywh2xyxy([x, y, w, h])
            input_boxes = np.array([xyxy])

            grounding_ret = {
                "image_source": image_source,
                "frame_idx": 0,
                "input_boxes": input_boxes,
                "labels": [caption_name],
            }

            if start_idx == end_idx:
                # appear for one frame
                masks, scores, logits = self.sam2_tracker.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes[None,:],
                    multimask_output = False,
                )
                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                tracking_combination[start_idx] = {1: masks>0.0}
                tracking_logits[start_idx] = {1:masks}

            else:
                # to select related bbox info
                bbox_first_idx = bbox_idx2frame[start_idx]
                bbox_end_idx = bbox_idx2frame[end_idx]

                # copy related images to a new dir
                current_dir = os.path.join(init_state_video_dir, caption_name, f"{segment_idx:05d}_[{start_idx}_{end_idx}]")

                print("[INFO] current_dir", current_dir)

                if os.path.exists(current_dir):
                    shutil.rmtree(current_dir)
                os.makedirs(current_dir, exist_ok=True)

                sub_remapping = {}
                for idx, frame_name in enumerate(frame_names[start_idx:end_idx+1]):
                    frame_name_mapping[frame_name] = (current_dir,f"{idx:05d}.jpg")
                    sub_remapping[idx] = frame_name

                    # copy files & rename
                    shutil.copyfile(
                        os.path.join(init_state_video_dir, frame_name),
                        os.path.join(current_dir, f"{idx:05d}.jpg"),
                    )

                    expect_bbox_idx = frame2idx[frame_name]

                    if expect_bbox_idx in bbox_idx2frame:
                        bbox_idx = bbox_idx2frame[expect_bbox_idx]
                        bbox_info[bbox_idx]["segment_frame_idx"] = idx

                        print("bbox_idx,bbox_first_idx,idx", bbox_idx, bbox_first_idx, idx)

                        assert bbox_info[bbox_idx]["frame_number"] == frame2idx[frame_name]
                    else:
                        print('[INFO] Relaxing...')
                        assert (expect_bbox_idx-1) in bbox_idx2frame or (expect_bbox_idx+1) in bbox_idx2frame

                sub_video_tracking_ret = self.sam2_tracker.tracking(
                    save_source_frames_segments_dir=current_dir,
                    grounding_ret=grounding_ret,
                    prompt_type_for_video=prompt_type_for_video,
                    all_bbox_info=bbox_info[bbox_first_idx : bbox_end_idx + 1],
                    zero_ann_frame_idx=True,
                    under_bbox_region=under_bbox_region,
                )
                for out_frame_idx, annots in sub_video_tracking_ret["tracking"].items():
                    frame_name = sub_remapping[out_frame_idx]
                    true_out_frame_idx = frame2idx[frame_name] # for source frames
                    tracking_combination[true_out_frame_idx] = annots
                    tracking_logits[true_out_frame_idx] = sub_video_tracking_ret["tracking_logits"][out_frame_idx]
                shutil.rmtree(current_dir)

        video_tracking_ret = {
            "grounding": {"labels":[caption_name]},
            "tracking": tracking_combination,
            "tracking_logits": tracking_logits,
            "height": height,
            "width": width,
        }
        return video_tracking_ret

    def visualize_tracking(
            self, 
            tracking_results_dir: str, 
            source_frame_dir: str, 
            video_segments: List,
            objects: List,
            output_video_path: str = None
        ):

        if not os.path.exists(tracking_results_dir):
            os.makedirs(tracking_results_dir)

        frame_names = get_all_frame_names(source_frame_dir)
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(objects, start=1)}
        for frame_idx, segments in video_segments.items():
            img = cv2.imread(os.path.join(source_frame_dir, frame_names[frame_idx]))

            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(
                scene=img.copy(), detections=detections
            )
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                annotated_frame,
                detections=detections,
                labels=[ID_TO_OBJECTS[i] for i in object_ids],
            )
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            cv2.imwrite(
                os.path.join(
                    tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"
                ),
                annotated_frame,
            )
        for frame_idx in range(len(frame_names)):
            if frame_idx not in video_segments:
                shutil.copyfile(
                    os.path.join(source_frame_dir, frame_names[frame_idx]),
                    os.path.join(
                        tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"
                    ),
                )

        if output_video_path is not None:
            create_video_from_images(
                tracking_results_dir,
                output_video_path,
                fps=25,
                compress_rate=5,
            )

    def tracking(
        self,
        video_path: str,
        save_source_frames_dir: str,
        tracking_results_dir: str,
        output_video_path: str,
        caption_name: str,
        video_stride: int = 1,
        start: int = 0,
        end: int = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        prompt_type_for_video: str = "box",
        only_with_caption: bool = True,
        initial_bbox_info: Optional[Dict] = None,
        all_bbox_info: Optional[List] = None,
        tracking_with_bbox_occurrence: bool = True,
        under_bbox_region: bool = False,
        mask_other_regions: bool = False,
        verbose: bool = True,
        save_tracking_results: bool = True,
        select_one_highest_conf: bool = False,
        grounding_with_max_confidence: bool = True,
    ):
        output_tracking_json_results = os.path.join(
                tracking_results_dir, "tracking_results_maskUtils_rle.json")
        output_tracking_bbox_json_results = os.path.join(
                tracking_results_dir, f"tracking_results_bbox.json"
        )
        if os.path.exists(output_tracking_json_results) and os.path.exists(output_tracking_bbox_json_results):
            print("[INFO] SKIP ONE! WAITING for NEXT!")
            return None
        
        assert all_bbox_info is not None and tracking_with_bbox_occurrence

        
        tracking_ret = self.tracking_video_with_occurrence(
            video_path=video_path,
            save_source_frames_dir=save_source_frames_dir,
            caption_name=caption_name,
            video_stride=video_stride,
            start=start,
            end=end,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            prompt_type_for_video=prompt_type_for_video,
            only_with_caption=only_with_caption,
            initial_bbox_info=initial_bbox_info,
            bbox_info=all_bbox_info,
            under_bbox_region=under_bbox_region,
            mask_other_regions=mask_other_regions,
            select_one_highest_conf=select_one_highest_conf,
            grounding_with_max_confidence=grounding_with_max_confidence,
        )
        
        if tracking_ret is None:
            print("[INFO] SKIP ONE! WAITING for NEXT!")
            return None

        if verbose:
            self.visualize_tracking(
                tracking_results_dir=tracking_results_dir,
                source_frame_dir=save_source_frames_dir,
                video_segments=tracking_ret["tracking"],
                objects=tracking_ret["grounding"]["labels"],
                output_video_path=output_video_path,
            )

        if save_tracking_results and tracking_ret is not None:
            os.makedirs(tracking_results_dir, exist_ok=True)
            
            new_tracking = {}
            for key, value in tracking_ret["tracking"].items():
                # key is the frame number
                # value: {"0": mask}
                assert len(value) == 1
                sub_k = list(value.keys())[0]
                mask_ = value[sub_k][0]

                if mask_.sum() > 0:
                    mask_segm = maskUtils.encode(np.asfortranarray(mask_))
                    mask_segm["counts"] = mask_segm["counts"].decode()
                    new_tracking[key] = mask_segm

            output_tracking_json_results = os.path.join(
                tracking_results_dir, "tracking_results_maskUtils_rle.json")
            with open(output_tracking_json_results, "w") as f:
                json.dump(new_tracking, f)
            print("Saving to tracking result to", output_tracking_json_results)

            bbox_tracking = []
            for key, value in tracking_ret["tracking"].items():
                # only support one tracking results for one item
                for k, v in value.items():

                    if v.sum() == 0:
                        continue
                    else:
                        xyxy = mask2bbox(v[0])
                        x,y,w,h = convert_xyxy2xywh(xyxy)

                        tmp_dict = {
                            "frame_number": key,
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h
                        }
                        bbox_tracking.append(tmp_dict)
                    break
            output_tracking_bbox_json_results = os.path.join(
                tracking_results_dir, f"tracking_results_bbox.json"
            )
            bbox_tracking.sort(key=lambda x:x["frame_number"])
            with open(output_tracking_bbox_json_results, "w") as f:
                json.dump(bbox_tracking, f)
            print("Saving to tracking result to", output_tracking_bbox_json_results)

        print("[INFO] FINISH ONE! WAITING for NEXT!")
        return tracking_ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--clip_fps", type=float, default=5.0)
    parser.add_argument("--video_stride", type=int, default=6)
    parser.add_argument("--prompt_type_for_video", type=str, default="mask")
    parser.add_argument(
        "--source_video_dir",
        type=str,
        default="ego4d_data/v2/clips/",
    )
    parser.add_argument("--clip_uid", type=str, default=None)
    parser.add_argument("--clip_uid_file", type=str, default=None)
    parser.add_argument("--bbox_file", type=str, default=None)

    parser.add_argument("--video_path", type=str, default="")

    parser.add_argument("--object_captions", nargs="+", type=str)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data_prepare/asset",
    )

    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="The frame number of the initial frame",
    )
    parser.add_argument(
        "--num_frames", type=int, default=0, help="To get the end frame number"
    )
    parser.add_argument(
        "--with_initial_bbox", action="store_true", help="whether to use the bbox information"
    )
    parser.add_argument(
        "--mask_other_regions", action="store_true", help="whether to use the bbox information (mask other region)"
    )
    parser.add_argument(
        "--under_bbox_region", action="store_true", help="whether to use bbox to refine the tracking results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to see the tracking video"
    )
    parser.add_argument(
        "--save_tracking_results", action="store_true", help="whether to save the tracking (mask) results"
    )
    parser.add_argument(
        "--grounding_with_max_confidence", action="store_true", help="choose which gorunding results for video tracking.)"
    )
    parser.add_argument(
        "--select_one_highest_conf", action="store_true", help="whether only to kept the highest grounding results. (Only valid for those using grounding dino.)"
    )
    parser.add_argument(
        "--only_with_caption", action="store_true", help="Ignore bbox information"
    )
    parser.add_argument(
        "--only_with_initial_bbox", action="store_true", help="Ignore other bbox information"
    )
    parser.add_argument(
        "--tracking_with_bbox_occurrence", action="store_true", help="Base on bbox from egotracks."
    )
    parser.add_argument(
        "--clip_with_object_caption_file", default="", help="JSON file than contains the clip_uid with corresponding object label to be tracked."
    )

    args = parser.parse_args()

    return args


def initializer_tracker_model():
    sam2_checkpoint = SAM2_1_CHECKPOINT
    sam2_config = "/" + os.path.join(SAM2_1_CONFIG_DIR, "sam2.1_hiera_l.yaml")
    print("[INFO] sam2_config", sam2_config)

    video_tracker = Tracker(
        grounding_model_config=GROUNDING_DINO_CONFIG,
        grounding_model_checkpoint=GROUNDING_DINO_CHECKPOINT,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return video_tracker

def one_sample_test():
    args = parse_args()
    print(args)
    save_source_frames_dir = os.path.join(args.save_dir, "source_frames")
    annoatated_frames_dir = os.path.join(args.save_dir, "annotated_frames")
    video_tracker = initializer_tracker_model()

    
        
    # clip_uid
    clip_data = read_txt(args.clip_uid_file)
    
    # provided bounding box information (if any)
    clip_bbox_data = {}
    if os.path.exists(args.bbox_file):
        clip_bbox_data = read_json(args.bbox_file)

    def process_one(clip_uid, box_info):
        video_path = os.path.join(args.source_video_dir, clip_uid + ".mp4")

        for object_caption in box_info.keys():
            caption = f"{object_caption}."
            caption_name = object_caption

            clip_save_source_frames_dir = os.path.join(
                save_source_frames_dir, f"{clip_uid}"
            )
            clip_tracking_results_dir = os.path.join(
                annoatated_frames_dir, f"{clip_uid}/{caption_name}"
            )
            output_video_path = os.path.join(
                clip_tracking_results_dir, "tracking_video.mp4"
            )

            tracking_ret = video_tracker.tracking(
                video_path=video_path,
                save_source_frames_dir=clip_save_source_frames_dir,
                tracking_results_dir=clip_tracking_results_dir,
                output_video_path=output_video_path,
                caption_name=caption,
                video_stride=args.video_stride,
                start=0,
                end=None,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                prompt_type_for_video=args.prompt_type_for_video,
                only_with_caption=args.only_with_caption,
                initial_bbox_info=(
                    None
                    if (
                        args.only_with_caption
                        or box_info is None
                        or not args.with_initial_bbox
                    )
                    else box_info[object_caption]["bbox_anno"][0]
                ),
                all_bbox_info=(
                    None
                    if (
                        args.only_with_caption
                        or box_info is None
                        or args.only_with_initial_bbox
                    )
                    else box_info[object_caption]["bbox_anno"]
                ),
                mask_other_regions=args.mask_other_regions and box_info is not None,
                tracking_with_bbox_occurrence=(
                    False
                    if (args.only_with_caption or box_info is None)
                    else args.tracking_with_bbox_occurrence
                ),
                under_bbox_region=args.under_bbox_region and box_info is not None,
                verbose=args.verbose,
                save_tracking_results=True,
                select_one_highest_conf=args.select_one_highest_conf,
                grounding_with_max_confidence=args.grounding_with_max_confidence,
            )
            if os.path.exists(clip_save_source_frames_dir):
                print("[INFO] Removing source files", clip_save_source_frames_dir)
                shutil.rmtree(clip_save_source_frames_dir)
            
    if args.clip_uid is not None:
        print("[INFO] Process one!")
        if args.clip_uid in clip_bbox_data:
            bbox_info = clip_bbox_data[args.clip_uid]
        else:
            bbox_info = None
        process_one(args.clip_uid, bbox_info)

    else:
        for clip_uid in clip_data:
            print(f"[INFO] Processing {clip_uid}")
            if clip_uid in clip_bbox_data:
                box_info = clip_bbox_data[clip_uid]
            else:
                bbox_info = None
            process_one(clip_uid, box_info)




if __name__ == "__main__":
    
    # Example
    # CUDA_VISIBLE_DEVICES=0 python preprocess/tracking/object_tracker.py --clip_uid_file sample/selected_clip_uids.txt --clip_uid 77e799b5-b99f-4e0d-abff-8baee6f76f82 --bbox_file sample/selected_egotracks_bboxs.json --tracking_with_bbox_occurrence --save_dir sample/sample_test --select_one_highest_conf --verbose 

    import time
    start_ = time.time()
    one_sample_test()
    print("[INFO] Time pass", time.time()-start_)
