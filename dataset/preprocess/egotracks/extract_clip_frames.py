import os
import cv2
from pathlib import Path
import supervision as sv
from tqdm import tqdm
import shutil


def extract_clip_frames(
    video_path: str,
    save_dir: str,
):
    save_source_frames_dir = os.path.join(save_dir, "source_frames")

    source_frames = Path(save_source_frames_dir)

    os.makedirs(save_source_frames_dir, exist_ok=True)
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    print(video_info)
    
    # Ego4D clip: 30FPS; egotracks: 5FPS
    frame_generator = sv.get_video_frames_generator(
        video_path, stride=6
    )

    source_frames.mkdir(parents=True, exist_ok=True)
    
    kept_frame_list = []
    with sv.ImageSink(
        target_dir_path=source_frames,
        overwrite=True,
        image_name_pattern="{:05d}.jpg",
    ) as sink:
        for frame_idx, frame in enumerate(tqdm(frame_generator, desc="Saving Video Frames")):
            sink.save_image(frame)
            if frame_idx % 5 == 0:
                current_fname = f"{frame_idx:05d}.jpg"
                kept_frame_list.append(current_fname)
            
    # we only need 1FPS
    for fname in kept_frame_list:
        src_fname = os.path.join(save_source_frames_dir, fname)
        tgt_fname = os.path.join(save_dir, fname)
        shutil.move(src_fname, tgt_fname)
    
    shutil.rmtree(source_frames)
    print(f"[INFO] Extracted frames saved to {save_dir}\n")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video_uid_file", type=str, required=True, help="Path to the video uid file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to video")
    parser.add_argument("--out_dir", type=str, default="./JPEGImages", help="Directory to save extracted frames")
    
    args = parser.parse_args()
    
    video_uids = []
    with open(args.video_uid_file, 'r') as f:
        for line in f:
            video_uid = line.strip()
            if not video_uid:
                continue
            video_uids.append(video_uid)
    
    print(f"[INFO] Found {len(video_uids)} video UIDs to process.")
    
    for video_uid in video_uids:
        
        video_path = os.path.join(args.data_dir, f"{video_uid}.mp4")
        save_dir = os.path.join(args.out_dir, video_uid)
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file {video_path} does not exist.")
            continue
        
        extract_clip_frames(
            video_path=video_path,
            save_dir=save_dir,
        )