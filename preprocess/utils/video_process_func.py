import os
import cv2
from pathlib import Path
import supervision as sv
import shutil
from typing import List, Literal,Optional, Dict
import numpy as np
from PIL import Image
import imageio.v3 as iio
import argparse
from tqdm import tqdm
import json
from .image_process_func import egotracks_xywh2xyxy

# this function is to extract valid filenames, given the dirpath
def get_all_frame_names(dirpath: str):
    frame_names = [
        p
        for p in os.listdir(dirpath)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].replace("img","")))
    return frame_names


def mask_other_region(
    source_frame_dir: str,
    tmp_source_frame_dir: str,
    all_bbox_info:List
):
    out_frame_bbox_info = {
        int(bbox["frame_number"]): egotracks_xywh2xyxy([bbox[k] for k in ["x", "y", "width", "height"]]) for bbox in all_bbox_info
    }

    if not os.path.exists(tmp_source_frame_dir):
        os.makedirs(tmp_source_frame_dir,exist_ok=True)
    frame_names = get_all_frame_names(source_frame_dir)

    # empty image
    img = cv2.imread(os.path.join(source_frame_dir, frame_names[0]))
    height,width,_ = img.shape
    empty_image = np.zeros_like(img) + 255

    for frame_idx, bbox_info in out_frame_bbox_info.items():
        img = cv2.imread(os.path.join(source_frame_dir, frame_names[frame_idx]))
        # height,width,_ = img.shape
        # mask
        x_min, y_min, x_max, y_max = bbox_info
        x_min, y_min, x_max, y_max = (
            max(round(x_min)-20,0),
            max(round(y_min)-20,0),
            min(round(x_max)+20,width),
            min(round(y_max)+20,height),
        )

        masked_part = np.zeros_like(img)
        masked_part[y_min:y_max, x_min:x_max,:] = 1
        masked_frame = np.where(masked_part, img, 255)

        cv2.imwrite(
            os.path.join(
                tmp_source_frame_dir, f"{frame_idx:05d}.jpg"
            ),
            masked_frame,
        )

    for frame_idx in range(len(frame_names)):
        if frame_idx not in out_frame_bbox_info:
            cv2.imwrite(
            os.path.join(
                tmp_source_frame_dir, f"{frame_idx:05d}.jpg"
            ),
            empty_image,
        )


# read video
def preprocess_video(
    video_path: str,
    save_source_frames_dir: str,
    video_stride: int = 1,
    start: int = 0,
    end: int = None,
    force_rewrite: bool = False,
):
    source_frames = Path(save_source_frames_dir)

    write_flag = (
        (
            os.path.exists(source_frames)
            and os.path.exists(os.path.join(save_source_frames_dir, "00000.jpg"))
            and force_rewrite
        )
        or (not os.path.exists(source_frames))
        or (not os.path.exists(os.path.join(save_source_frames_dir, "00000.jpg")))
    )

    os.makedirs(save_source_frames_dir, exist_ok=True)
    if write_flag:
        print("[Rewriting]")
        shutil.rmtree(source_frames)
        video_info = sv.VideoInfo.from_video_path(video_path)
        print(video_info)
        frame_generator = sv.get_video_frames_generator(
            video_path, stride=video_stride, start=start, end=end
        )

        source_frames.mkdir(parents=True, exist_ok=True)
        with sv.ImageSink(
            target_dir_path=source_frames,
            overwrite=True,
            image_name_pattern="{:05d}.jpg",
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

    frame_names = get_all_frame_names(save_source_frames_dir)
    return frame_names


# read video
def preprocess_video_after_selection(
    video_path: str,
    save_source_frames_dir: str,
    video_stride: int = 1,
    start: int = 0,
    end: int = None,
    force_rewrite: bool = False,
):
    source_frames = Path(save_source_frames_dir)

    write_flag = (
        (
            os.path.exists(source_frames)
            and os.path.exists(os.path.join(save_source_frames_dir, "00000.jpg"))
            and force_rewrite
        )
        or (not os.path.exists(source_frames))
        or (not os.path.exists(os.path.join(save_source_frames_dir, "00000.jpg")))
    )

    os.makedirs(save_source_frames_dir, exist_ok=True)
    if write_flag:
        print("[Rewriting]")
        shutil.rmtree(source_frames)
        video_info = sv.VideoInfo.from_video_path(video_path)
        print(video_info)
        frame_generator = sv.get_video_frames_generator(
            video_path, stride=video_stride
        )

        source_frames.mkdir(parents=True, exist_ok=True)
        with sv.ImageSink(
            target_dir_path=source_frames,
            overwrite=True,
            image_name_pattern="{:05d}.jpg",
        ) as sink:
            for idx, frame in enumerate(tqdm(frame_generator, desc="Saving Video Frames")):
                # [start, end)
                if end is not None and idx >= end:
                    break
                if idx < start:
                    continue
                sink.save_image(frame)

    frame_names = get_all_frame_names(save_source_frames_dir)
    return frame_names


def create_video_from_images(image_folder, output_video_path, fps=25, compress_rate=1):
    if output_video_path.endswith("mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif output_video_path.endswith("avi"):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif output_video_path.endswith("flv"):
        fourcc = cv2.VideoWriter_fourcc(*"FLV1")
    elif output_video_path.endswith("ogv"):
        fourcc = cv2.VideoWriter_fourcc(*"THEO")
    else:
        raise NotImplementedError("Unsupported output format!")

    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ]
    image_files.sort()  # sort the files in alphabetical order
    print(
        "[INFO] Image: {} ~ {} (total {} images)".format(
            image_files[0], image_files[-1], len(image_files)
        )
    )
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    # first_image:
    image_path = os.path.join(image_folder, image_files[0])
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    h, w = h // compress_rate, w // compress_rate

    out_video = cv2.VideoWriter(
        output_video_path, fourcc, fps, frameSize=(w, h), isColor=True
    )
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (w, h))
        out_video.write(img)

    if out_video is not None and out_video.isOpened():
        out_video.release()
        print(f"Video saved at {output_video_path}")
    else:
        print("[ERROR] Fail to save video!")


def create_video_from_image_files(image_files, output_video_path, fps=25, compress_rate=1):
    if output_video_path.endswith("mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif output_video_path.endswith("avi"):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    elif output_video_path.endswith("flv"):
        fourcc = cv2.VideoWriter_fourcc(*"FLV1")
    elif output_video_path.endswith("ogv"):
        fourcc = cv2.VideoWriter_fourcc(*"THEO")
    else:
        raise NotImplementedError("Unsupported output format!")

    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    
    image_files.sort()  # sort the files in alphabetical order
    print(
        "[INFO] Image: {} ~ {} (total {} images)".format(
            image_files[0], image_files[-1], len(image_files)
        )
    )
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    # first_image:
    image_path = image_files[0]
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    h, w = h // compress_rate, w // compress_rate

    out_video = cv2.VideoWriter(
        output_video_path, fourcc, fps, frameSize=(w, h), isColor=True
    )
    for image_path in tqdm(image_files):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (w, h))
        out_video.write(img)

    if out_video is not None and out_video.isOpened():
        out_video.release()
        print(f"Video saved at {output_video_path}")
    else:
        print("[ERROR] Fail to save video!")


# write to gif

def create_gif_from_images(image_folder, output_gif_path, fps=25, compress_rate=4):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

    # get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1] in valid_extensions
    ]
    image_files.sort()  # sort the files in alphabetical order
    print(
        "[INFO] Image: {} ~ {} (total {} images)".format(
            image_files[0], image_files[-1], len(image_files)
        )
    )
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # first_image:
    image_path = os.path.join(image_folder, image_files[0])
    img = iio.imread(image_path)
    h, w, _ = img.shape

    gif_images = []
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        img = iio.imread(image_path)
        img = cv2.resize(img, (w // compress_rate, h // compress_rate))
        gif_images.append(img)

    # source release
    iio.imwrite(output_gif_path, gif_images, fps=fps)
    print(f"Gif saved at {output_gif_path}")
