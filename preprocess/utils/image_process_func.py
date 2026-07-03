import os
import json
import numpy as np
from typing import List
import base64
from PIL import Image
from io import BytesIO
import cv2
import pycocotools.mask as maskUtils

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def build_base64_frames(source_frame_dir, frame_names, selected_frames_idx,verbose=False, output_dir="tmp"):
    base64_frames = []
    for idx, frame_idx in enumerate(selected_frames_idx):
        frame = cv2.imread(os.path.join(source_frame_dir, frame_names[frame_idx]))
        
        if verbose:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"{idx}_clip_frame_{frame_idx}.jpg"), frame)

        
        frame_rgb = Image.fromarray(frame).convert("RGB")
        buffered = BytesIO()
        frame_rgb.save(buffered, format="JPEG")
        base64_frame = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_frames.append(base64_frame)
    return base64_frames

def build_base64_frames_from_np(np_frames):
    base64_frames = []
    buffered = BytesIO()
    for frame in np_frames:
        frame_rgb = Image.fromarray(frame).convert("RGB")
        buffered = BytesIO()
        frame_rgb.save(buffered, format="JPEG")
        base64_frame = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_frames.append(base64_frame)
    return base64_frames


###############################
# Annotation Related (mask)
###############################

def mask2rle(img):
    """
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(256, 256)):
    if mask_rle != "None":
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T
    else:
        return np.zeros([shape[1], shape[0]], dtype=np.uint8)

def convert_rle_file_to_mask(file_name):
    with open(file_name,"r") as f:
        rle_data = json.load(f)

    h, w = file_name.split("/")[-1].split(".")[0].split("_hw_")[-1].split("_")
    h, w = int(h), int(w)
    mask_data = {}

    for out_frame_idx, annots in rle_data.items():
        mask_data[out_frame_idx] = {}
        for obj_id, obj_anno_list in annots.items():
            mask_data[out_frame_idx][obj_id] = []
            for obj_anno in obj_anno_list:
                mask_ = rle2mask(obj_anno, shape=(w, h))
                mask_data[out_frame_idx][obj_id].append(mask_)

    return mask_data

def mask2bbox(mask):
    if mask.sum() == 0:
        return [0, 0, 0, 0]
    nonzero_indices = np.nonzero(mask)
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    bbox = [int(min_x), int(min_y), int(max_x),int(max_y)]
    return bbox


'''
Annotation Related (mask)
'''
def convert_xyxy2xywh(xyxy):
    min_x, min_y, max_x, max_y = xyxy
    w = max_x - min_x
    h = max_y - min_y
    return [min_x, min_y, w, h]

def egotracks_xywh2xyxy(xywh):
    if type(xywh) is dict:
        x,y,w,h = [xywh[k] for k in ["x","y","width","height"]]
    else:
        x, y, w, h = xywh
    new_x, new_y, new_w, new_h = x, y, x + w, y + h
    return [new_x, new_y, new_w, new_h]

# BGR in, BGR out
def draw_box(
    frame: np.ndarray,
    boxes: List[List[float]],
    color: int = "red",
    label: str = None,
    RGB_in: bool = False,
    RGB_out: bool = False,
):
    import supervision as sv

    colormap = {
        "red": sv.Color.RED,
        "blue": sv.Color.BLUE,
        "green": sv.Color.GREEN,
        "yellow": sv.Color.YELLOW,
        "white": sv.Color.WHITE,
        "black": sv.Color.BLACK,
        "roboflow": sv.Color.ROBOFLOW,
    }

    boxes = np.array(boxes)
    # assume all boxes are of different categories
    detections = sv.Detections(xyxy=boxes, class_id=np.arange(len(boxes)))

    box_annotator = sv.BoxAnnotator(
        colormap[color],
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator()

    if RGB_in:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        frame,
        detections=detections,
        labels=label,
    )
    if RGB_out:
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame


# mask related
def convert_mask_dir_to_mask(mask_dir):
    ret = {}
    fn_list = os.listdir(mask_dir)
    fn_list.sort()

    for fn in fn_list:
        frame_mask = np.array(Image.open(os.path.join(mask_dir, fn)))
        frame_idx = int(fn.split(".")[0])
        # frame_mask = np.where(frame_mask==1,1,0).astype(np.uint8)
        mask_segm = maskUtils.encode(np.asfortranarray(frame_mask))
        mask_segm["counts"] = mask_segm["counts"].decode()
        ret[str(frame_idx)] = mask_segm
    return ret
