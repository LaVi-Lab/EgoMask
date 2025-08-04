path_to_ego4d="tmp/ego4d"
aws_profile_name="ego4d" # or default

################################
# Download Ego4D clips
################################
echo "Downloading Ego4D clips for training..."

ego4d --output_directory ${path_to_ego4d}  --dataset clips --video_uid_file  preprocess/egotracks/traingingdata_video_ids.txt --aws_profile_name ${aws_profile_name}

################################
# Extract frames
################################
echo "Extracting frames from Ego4D clips..."

python preprocess/egotracks/extract_clip_frames.py --data_dir ${path_to_ego4d} --video_uid_file preprocess/egotracks/traingingdata_video_ids.txt --out_dir egomask-train/train/JPEGImages

################################
# rm tmp directory
################################
rm -rf ${path_to_ego4d}