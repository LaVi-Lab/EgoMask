path_to_ego4d="tmp/ego4d"
path_to_fps2_images="tmp/fps2_images"
aws_profile_name="ego4d" # or default

################################
# Download Ego4D clips
################################
echo "Downloading Ego4D full scale videos for RefEgo..."

ego4d --output_directory ${path_to_ego4d} --dataset full_scale --video_uid_file preprocess/refego/video_list.txt  --version v1 --aws_profile_name ${aws_profile_name}

################################
# Apply ffmpeg for Ego4D v1 videos. 
################################
echo "Applying ffmpeg to Ego4D full scale videos..."

bash preprocess/refego/extract_images.sh [path_to_ego4d]/full_scale ${path_to_fps2_images}

################################
# Split images into RefEgo style video clips
################################
echo "Splitting images into RefEgo style video clips..."

bash preprocess/refego/split_video_clips.sh ${path_to_fps2_images} egomask/JPEGImages/refego

################################
# rm tmp directory
################################
rm -rf ${path_to_fps2_images}
rm -rf ${path_to_ego4d}