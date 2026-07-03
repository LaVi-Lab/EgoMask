import os

PROJECT_PATH = "./"  # PATH TO THIS PROJECT
GROUNDED_SAM_2_DIR = ""  # PATH TO GROUNDED_SAM_2 REPOSITORY
GROUNDING_DINO_CONFIG = os.path.join(
        GROUNDED_SAM_2_DIR,
        "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
)
GROUNDING_DINO_CHECKPOINT = os.path.join(
        GROUNDED_SAM_2_DIR, "gdino_checkpoints/groundingdino_swint_ogc.pth"
)
SAM2_1_CONFIG_DIR = os.path.join(GROUNDED_SAM_2_DIR, "sam2/configs/sam2.1") 
SAM2_1_CHEKPOINT_DIR = "facebook/sam2.1/"
SAM2_1_CHECKPOINT = SAM2_1_CHEKPOINT_DIR + "sam2.1_hiera_large.pt"

EGO4D_FILE = "ego4d_data/ego4d.json"
EGOTRACKS_DIR = r"ego4d_data/v2/egotracks"
RAW_NARRATION_FILE = "ego4d_data/v2/annotations/narration.json"

egotrakcs_bbox_filename = "egotracks_clip_object_bbox.json"
egotracks_clip2vide_filename = "egotracks_clip2video_uid.json"
ego4d_clip2video_filename = "ego4d_clip2video_uid.json"

SOURCE_FRAME_DIR = "source_frame"
ANNOTATED_FRAME_DIR = "annotated_frame"
