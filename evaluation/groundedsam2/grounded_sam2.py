import os
import sys

import json
import torch
import shutil
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import pycocotools.mask as maskUtils
from torchvision.ops import box_convert
from evaluation.common_utils import get_all_frame_names, write_json


from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.track_utils import sample_points_from_masks
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def convert_single_format_for_save(grounding_ret):
    image_source = grounding_ret["image_source"]
    height, width, _ = image_source.shape
    boxes = grounding_ret["boxes"] * torch.Tensor([width, height, width, height])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
    
    confidences = grounding_ret["confidences"].numpy().tolist()
        

    ret_value = {
        "frame_idx": grounding_ret["frame_idx"],
        "frame_name": grounding_ret["frame_name"],
        "input_boxes": input_boxes,
        "confidences": confidences,
        "labels": grounding_ret["labels"],
    }
    return ret_value

class GroundingDino_model:
    def __init__(
        self,
        grounding_model_config: str,
        grounding_model_checkpoint: str,
        device: str = "cuda",
    ):
        self.grounding_model = load_model(
            model_config_path=grounding_model_config,
            model_checkpoint_path=grounding_model_checkpoint,
            device=device,
        )
        self.grounding_model.eval()
    
    def prompt_grounding_model(
        self,
        save_dir,
        frame_names,
        caption,
        box_threshold,
        text_threshold,
        grounding_with_max_confidence: bool = True,
    ):
        """
        save_dir: str, where to put the frames from the video
        frame_names: List[str], frame filepath
        caption: str
        box_threshold: float
        text_threshold: float
        """
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
                    "frame_name": frame_name.replace(".jpg",""),
                }
                if max(confidences) > max_confidence:
                    kept_result = ret
                    max_confidence = max(confidences)

                if not grounding_with_max_confidence:
                    print("[INFO] GroundingDino first find object at Frame:", idx)
                    return ret

        if kept_result:
            idx = kept_result["frame_idx"]
            print("[INFO] GroundingDino find object with max_confidence at Frame:", idx)
            return kept_result
        print("[WARN] GroundingDino cannot find object in the video")
        return None

    def convert_return_format(self, grounding_ret, select_one=True):
        image_source = grounding_ret["image_source"]
        ann_frame_idx = grounding_ret["frame_idx"]
        height, width, _ = image_source.shape
        boxes = grounding_ret["boxes"] * torch.Tensor([width, height, width, height])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        objects = grounding_ret["labels"]

        if select_one and len(objects) > 1:
            confidences = grounding_ret["confidences"].numpy().tolist()
            confidences_with_idx = [(idx, conf) for idx, conf in enumerate(confidences)]
            confidences_with_idx.sort(key=lambda x: -x[-1])
            max_conf_idx = confidences_with_idx[0][0]
            input_boxes = [input_boxes[max_conf_idx]]
            objects = [objects[max_conf_idx]]

        ret_value = {
            "image_source": image_source,
            "frame_idx": ann_frame_idx,
            "frame_name": grounding_ret["frame_name"],
            "input_boxes": input_boxes,
            "labels": objects,
        }
        return ret_value


    
class SAM2_Tracker:
    def __init__(self, sam2_config: str, 
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

    def tracking(self, 
        save_source_frames_segments_dir: str,
        grounding_ret: Dict = None,
        prompt_type_for_video: str = "box",
        zero_ann_frame_idx: bool = True):

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
            masks=torch.from_numpy(mask_ret["masks"]),
            inference_state=inference_state,
            ann_frame_idx=0 if zero_ann_frame_idx else grounding_ret["frame_idx"],
        )

        video_segments = {}

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
    def __init__(
        self,
        grounding_model_config: str,
        grounding_model_checkpoint: str,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str = "cuda",
    ):
        self.grounding_model = GroundingDino_model(
            grounding_model_config, grounding_model_checkpoint, device
        )

        self.sam2_tracker = SAM2_Tracker(sam2_config, sam2_checkpoint)

    def tracking_video(self, 
                    frame_names: List[str],
                    save_source_frames_dir: str,
                    caption_name: str,
                    box_threshold: float = 0.35,
                    text_threshold: float = 0.35,
                    prompt_type_for_video: str = "box",
                    select_one_highest_conf: bool = True,
                    grounding_with_max_confidence:bool=True): # False: navie baseline
        frame2idx = {frame_name: idx for idx, frame_name in enumerate(frame_names)}
        init_state_video_dir = save_source_frames_dir
        grounding_ret = self.grounding_model.prompt_grounding_model(
            save_dir=init_state_video_dir,
            frame_names=frame_names,
            caption=caption_name,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            grounding_with_max_confidence=grounding_with_max_confidence,
        )
        if grounding_ret is not None:
            grounding_ret = self.grounding_model.convert_return_format(grounding_ret,select_one=select_one_highest_conf)

        else:
            print("[WARN] No object found in the video")
            return None
        split_frame_idx = grounding_ret["frame_idx"]

        if split_frame_idx == 0:

            video_tracking_ret = self.sam2_tracker.tracking(
                save_source_frames_segments_dir=init_state_video_dir,
                grounding_ret=grounding_ret,
                prompt_type_for_video=prompt_type_for_video,
                zero_ann_frame_idx=True,
            )
        elif split_frame_idx > 0:
            # remove to different sub-frames
            if len(caption_name) > 60:
                caption_name = caption_name[:60]
            
            part1_dir = os.path.join(
                save_source_frames_dir, caption_name, f"reverse_[0_{split_frame_idx}]"
            )
            print("[INFO] part1_dir", part1_dir)
            if os.path.exists(part1_dir):
                shutil.rmtree(part1_dir)
            os.makedirs(part1_dir, exist_ok=True)

            frame_name_mapping = {}
            part1_remapping = {}
            part2_remapping = {}

            for idx, frame_name in enumerate(frame_names[:split_frame_idx+1][::-1]):
                frame_name_mapping[frame_name] = (part1_dir, f"{idx:05d}.jpg")
                part1_remapping[idx] = frame_name
                
                shutil.copyfile(
                    os.path.join(init_state_video_dir, frame_name),
                    os.path.join(part1_dir, f"{idx:05d}.jpg"),
                )

            max_frame_idx = len(frame_names)-1
            part2_dir = os.path.join(
                save_source_frames_dir,
                caption_name,
                f"[{split_frame_idx}_{max_frame_idx}]",
            )
            print("[INFO] part2_dir", part2_dir)

            if os.path.exists(part2_dir):
                shutil.rmtree(part2_dir)
            os.makedirs(part2_dir, exist_ok=True)

            for idx, frame_name in enumerate(frame_names[split_frame_idx:]):
                frame_name_mapping[frame_name] = (part2_dir,f"{idx:05d}.jpg",)
                part2_remapping[idx] = frame_name
                shutil.copyfile(
                    os.path.join(init_state_video_dir, frame_name),
                    os.path.join(part2_dir, f"{idx:05d}.jpg"),
                )

            # tracking
            part1_video_tracking_ret = self.sam2_tracker.tracking(
                save_source_frames_segments_dir=part1_dir,
                grounding_ret=grounding_ret,
                prompt_type_for_video=prompt_type_for_video,
                zero_ann_frame_idx=True,
            )

            part2_video_tracking_ret = self.sam2_tracker.tracking(
                save_source_frames_segments_dir=part2_dir,
                grounding_ret=grounding_ret,
                prompt_type_for_video=prompt_type_for_video,
                zero_ann_frame_idx=True,
            )
            tracking_combination = {}
            tracking_ret_list = [part1_video_tracking_ret, part2_video_tracking_ret]
            sub_remapping_list = [part1_remapping, part2_remapping]
            for tracking_ret, sub_remapping in zip(tracking_ret_list, sub_remapping_list):
                for out_frame_idx, annots in tracking_ret["tracking"].items():
                    frame_name = sub_remapping[out_frame_idx]
                    true_out_frame_idx = frame2idx[frame_name]
                    tracking_combination[true_out_frame_idx] = annots

            video_tracking_ret = {
                "grounding": grounding_ret,
                "initial_mask": part1_video_tracking_ret["initial_mask"],
                "tracking": tracking_combination,
                "height": part1_video_tracking_ret["height"],
                "width": part1_video_tracking_ret["width"],
            }

            shutil.rmtree(part1_dir)
            shutil.rmtree(part2_dir)
        return video_tracking_ret

    def tracking(self, 
                image_dir:str,
                frame_names: List[str],
                save_source_frames_dir: str, 
                tracking_results_dir: str,
                tracking_file_name:str,
                caption_name: str,
                box_threshold: float = 0.35,
                text_threshold: float = 0.35,
                prompt_type_for_video: str = "box",
                select_one_highest_conf: bool = True,
                grounding_with_max_confidence:bool=True):
        print("[TRACKING] image_dir", image_dir)

        os.makedirs(save_source_frames_dir, exist_ok=True)

        # extract images
        init_state_video_dir = image_dir
        all_frame_names = get_all_frame_names(init_state_video_dir)
        if all_frame_names != frame_names:
            # copy frame names to the save_source_frames_dir
            for frame_name in frame_names:
                shutil.copyfile(
                    os.path.join(init_state_video_dir, frame_name),
                    os.path.join(save_source_frames_dir, frame_name),
                )
            init_state_video_dir = save_source_frames_dir
        print('[INFO] init_state_video_dir', init_state_video_dir)
        tracking_ret = self.tracking_video(
            # image_dir=image_dir,
            frame_names=frame_names,
            save_source_frames_dir=init_state_video_dir,
            caption_name=caption_name,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            prompt_type_for_video=prompt_type_for_video,
            select_one_highest_conf=select_one_highest_conf,
            grounding_with_max_confidence=grounding_with_max_confidence,
        )

        new_tracking = {}

        os.makedirs(tracking_results_dir, exist_ok=True)

        if tracking_ret is not None:
            for key, value in tracking_ret["tracking"].items():
                assert len(value) == 1
                sub_k = list(value.keys())[0]
                mask_ = value[sub_k][0]
                file_key = frame_names[key].replace(".jpg", "")
                if mask_.sum() > 0:
                    mask_segm = maskUtils.encode(np.asfortranarray(mask_))
                    mask_segm["counts"] = mask_segm["counts"].decode()
                    new_tracking[file_key] = mask_segm

        output_tracking_json_results = os.path.join(
            tracking_results_dir, tracking_file_name
        )

        write_json(new_tracking, output_tracking_json_results)

        print("Saving to tracking result to", output_tracking_json_results)

        shutil.rmtree(save_source_frames_dir, ignore_errors=True)
        
        # detection
        new_grounding_ret = {}
        if tracking_ret is  None:
            new_grounding_ret["frame_idx"] = -1
        else:
            new_grounding_ret["frame_idx"] = tracking_ret["grounding"]["frame_idx"]
            new_grounding_ret["frame_name"] = tracking_ret["grounding"]["frame_name"]
            new_grounding_ret["labels"] = tracking_ret["grounding"]["labels"]
            new_grounding_ret["input_boxes"] = tracking_ret["grounding"]["input_boxes"][0].tolist()
            write_json(new_grounding_ret,output_tracking_json_results.replace(".json", "_grounding_ret.json") )
        
        
        
        return new_tracking