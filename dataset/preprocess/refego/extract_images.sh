#!/bin/bash

# Reference: https://github.com/shuheikurita/RefEgo/blob/main/dataset/extract_images.sh

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 path_to_ego4d path_to_fps2 [--cuda]"
    exit 1
fi

path_to_ego4d=$1
path_to_fps2=$2
use_cuda=""

if [ "$#" -eq 3 ] && [ "$3" = "--cuda" ]; then
    use_cuda="-hwaccel cuda"
fi

total_lines=$(wc -l < "video_list.txt")
count=0

while IFS= read -r uid
do
    count=$((count + 1))
    mkdir -p "${path_to_fps2}/${uid}"
    echo ffmpeg $use_cuda -i "${path_to_ego4d}/ego4d/v1/full_scale/${uid}.mp4" -vf "fps=2,scale=640:-1" -qscale 1 "${path_to_fps2}/${uid}/img%10d.jpg"
    ffmpeg $use_cuda -i "${path_to_ego4d}/ego4d/v1/full_scale/${uid}.mp4" -vf "fps=2,scale=640:-1" -qscale 1 "${path_to_fps2}/${uid}/img%10d.jpg"

    progress=$(echo "scale=4; $count / $total_lines * 100" | bc)
    echo "Progress: $progress%"
done < "video_list.txt"