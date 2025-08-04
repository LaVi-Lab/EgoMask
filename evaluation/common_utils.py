import os
import json
import jsonlines
import numpy as np

root_dir = "dataset/egomask"


long_subset = {
    "meta_data_file": f"{root_dir}/subset/long/meta.json",
    "meta_expression_file": f"{root_dir}/subset/long/meta_expressions.json",
    "mask_dir": f"{root_dir}/annotations",
    "source_frames_dir": f"{root_dir}/JPEGImages/egotracks",
}

medium_subset = {
    "meta_data_file": f"{root_dir}/subset/long/meta.json",
    "meta_expression_file": f"{root_dir}/subset/medium/meta_expressions.json",
    "mask_dir": f"{root_dir}/annotations",
    "source_frames_dir": f"{root_dir}/JPEGImages/egotracks",
}

short_subset = {
    "meta_data_file": f"{root_dir}/subset/short/meta.json",
    "meta_expression_file": f"{root_dir}/subset/short/meta_expressions.json",
    "mask_dir": f"{root_dir}/annotations",
    "source_frames_dir": f"{root_dir}/JPEGImages/refego",
}

full_set = {
    "meta_data_file": f"{root_dir}/meta.json",
    "meta_expression_file": f"{root_dir}/meta_expressions.json",
    "mask_dir": f"{root_dir}/annotations",
    "source_frames_dir": f"{root_dir}/JPEGImages/egotracks",
}

dataset_mapping = {
    "long": long_subset,
    "medium": medium_subset,
    "short": short_subset,
    "full": full_set,
}

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_jsonl(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        data = [obj for obj in reader]
    return data

def write_jsonl(data, file_path):
    with jsonlines.open(file_path, 'w') as writer:
        for obj in data:
            writer.write(obj)



def get_all_frame_names(dirpath: str):
    frame_names = [
        p
        for p in os.listdir(dirpath)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(
        key=lambda p: int(
            os.path.splitext(p)[0]
            .replace("img", "")
            .replace("frame", "")
            .replace("ring_front_center_","")
        )
    )
    return frame_names

def mask2bbox(mask):
    # mask --> (x_min, y_min, x_max, y_max)
    if mask.sum() == 0:
        return [0, 0, 0, 0]
    nonzero_indices = np.nonzero(mask)
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    bbox = [int(min_x), int(min_y), int(max_x),int(max_y)]
    return bbox

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1 (tuple): A tuple of (x1, y1, x2, y2) for the first box.
    box2 (tuple): A tuple of (x1, y1, x2, y2) for the second box.

    Returns:
    float: The IoU of the two boxes.
    """

    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate the area of intersection rectangle
    inter_width = max(inter_x_max - inter_x_min, 0)
    inter_height = max(inter_y_max - inter_y_min, 0)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the area of union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou