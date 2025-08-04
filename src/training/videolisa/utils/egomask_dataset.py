"""
Modified from https://github.com/wjn922/ReferFormer/blob/main/datasets/ytvos.py
Ref-YoutubeVOS data loader
"""

import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import cv2
import json
import numpy as np
import random
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
from pycocotools import mask as maskUtils

class EgoMask_Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        num_frames_sparse=50,
        num_frames_dense=4,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = os.path.join(base_image_dir, "egomask-train")
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_frames_sparse = num_frames_sparse
        self.num_frames_dense = num_frames_dense

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.dataset_cls = EgoMaskDataset(img_folder=os.path.join(self.base_image_dir, "train"),
                                           ann_file=os.path.join(self.base_image_dir,
                                                                 "meta_expressions/train/meta_expressions.json"),
                                           num_frames=self.num_frames_sparse,
                                           min_valid_frames=self.num_frames_dense,
                                           target_img_size=self.img_size)
        print('Using EgoMask-Train')

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]

        # if h > self.img_size or w > self.img_size:
        #     new_height, new_width = get_resized_num(h, w, self.img_size)

        #     x = F.interpolate(
        #         x.unsqueeze(0),
        #         size=(new_height, new_width),
        #         mode="bilinear",
        #         align_corners=False,
        #     ).squeeze(0)
        #     h, w = resized_image.shape[-2:]
        # assert padh >= 0 and padw >= 0
        padh = self.img_size - h
        padw = self.img_size - w
        
        assert padh >= 0 and padw >= 0
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_dense_indices(self):
        sequence = np.arange(self.num_frames_sparse)
        random_numbers = np.random.choice(sequence, size=self.num_frames_dense, replace=False)

        return sorted(random_numbers.tolist())

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.dataset_cls) - 1)
        image_list, target = self.dataset_cls.__getitem__(idx)

        
        selected_index = range(len(target["selected_valid"]))
        dense_index = sorted(random.sample(selected_index, self.num_frames_dense))
        dense_indices = [target["selected_valid"][i] for i in dense_index]
        assert len(set(dense_indices)) == self.num_frames_dense
        

        # pre-process for CLIP
        image_clip_list = []
        for img in image_list:
            image_clip = self.clip_image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            image_clip_list.append(image_clip)
        video_data_clip_tsr = torch.stack(image_clip_list, dim=0)

        # pre-process for SAM
        image_sam_list = []
        for idx, image in enumerate(image_list):
            if idx in dense_indices:
                image_sam = self.transform.apply_image(image)
                image_sam_list.append(image_sam)
        resize = image_sam_list[0].shape[:2]

        mask_list = []
        for idx in range(target["masks"].shape[0]):
            if idx in dense_indices:
                mask_list.append(target["masks"][idx])
        masks = torch.stack(mask_list, dim=0)

        questions = []
        answers = []
        sampled_classes = [target["caption"]]
        for text in sampled_classes:
            text = text.strip()
            if text[-1] == ".":
                text = text[:-1]
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image_sam_list_proc = []
        for image in image_sam_list:
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            image_sam_list_proc.append(image)
        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        assert video_data_clip_tsr.shape[0] == self.num_frames_sparse
        if not (image_sam_tsr.shape[0] == masks.shape[0] == self.num_frames_dense):
            print(image_sam_tsr.shape[0], masks.shape[0], self.num_frames_dense)
            print(target["video"])
            print(target["caption"])
        assert image_sam_tsr.shape[0] == masks.shape[0] == self.num_frames_dense

        # print("questions", questions)
        return (
            None,
            image_sam_tsr,
            video_data_clip_tsr,
            conversations,
            masks,
            label,
            dense_indices,
            resize,
            questions,
            sampled_classes,
        )


class EgoMaskDataset(Dataset):

    def __init__(
        self, img_folder, ann_file, num_frames, min_valid_frames, target_img_size= None,
    ):
        self.img_folder = img_folder
        self.ann_file = ann_file

        self.subset = os.path.basename(img_folder)

        self.num_frames = num_frames

        self.min_valid_frames = min_valid_frames

        self.target_img_size = target_img_size
        self.prepare_metas()

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        
        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)

            for exp_id, exp_dict in vid_data['expressions'].items():
                cur_obj_id = exp_dict['obj_id']
                cur_obj_valid_frames = sorted(vid_meta["objects"][cur_obj_id]["frames"])

                cur_valid_frame_idxs = [int(frame_fn) for frame_fn in cur_obj_valid_frames]
                cur_obj_category = vid_meta["objects"][cur_obj_id]["category"]

                if self.subset == "valid":
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(cur_obj_id)
                    meta['frames'] = vid_frames 
                    meta['frame_id'] = frame_id
                    # where the objects shows
                    meta["valid_frames"] = cur_obj_valid_frames
                    # get object category
                    meta['category'] = cur_obj_category
                    self.metas.append(meta)

                else:
                    for frame_id in range(0, vid_len, self.num_frames):

                        # cur_valid_frame_idxs
                        relax_frames = list(range(max(0, frame_id - self.num_frames//2),min(frame_id + self.num_frames//2, vid_len)))

                        # make sure the object can appear
                        if len(set(relax_frames) & set(cur_valid_frame_idxs)) > 0:
                            meta = {}
                            meta['video'] = vid
                            meta['exp'] = exp_dict['exp']
                            meta['obj_id'] = int(exp_dict['obj_id'])
                            meta['frames'] = vid_frames
                            meta['frame_id'] = frame_id

                            # where the objects shows
                            meta["valid_frames"] = cur_obj_valid_frames
                            # get object category
                            meta['category'] = cur_obj_category
                            self.metas.append(meta)

    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]  # dict

        video, exp, obj_id, category, frames, frame_id = \
            meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
        # clean up the caption
        exp = " ".join(exp.lower().split())
        vid_len = len(frames)

        num_frames = self.num_frames

        valid_frames = meta["valid_frames"]
        valid_indx = [frames.index(fn) for fn in valid_frames]

        # random sparse sample
        sample_indx = [frame_id]
        if self.num_frames != 1:
            # local sample
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                global_n = num_frames - len(sample_indx)

                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >= global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    num_repeat = global_n // vid_len
                    select_id = random.sample(range(vid_len), global_n % vid_len) + list(range(vid_len)) * num_repeat
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                    assert len(sample_indx) == self.num_frames

        selected_valid = []

        if len(set(valid_indx) & set(sample_indx)) < self.min_valid_frames:

            selected_indx = random.sample(sample_indx, len(sample_indx)-self.min_valid_frames)

            remaining_ = self.min_valid_frames - len(valid_indx)
            if remaining_ > 0: # 4,2
                selected_valid = random.choices(valid_indx, k=self.min_valid_frames)                   
            elif remaining_ == 0:
                selected_valid = valid_indx
            elif remaining_ < 0:
                selected_valid = random.sample(valid_indx, self.min_valid_frames)

            sample_indx = selected_indx + selected_valid
        else:
            selected_valid = list(set(valid_indx) & set(sample_indx))

        non_valid = list(set(sample_indx) - set(valid_indx))
        non_valid = random.sample(non_valid, min(len(non_valid), self.min_valid_frames//2))

        assert len(selected_valid) >= self.min_valid_frames       

        assert len(sample_indx) == self.num_frames

        sample_indx.sort()      # ensure the video in correct temporal order

        # previous no non_valid, now add non_valid
        selected_valid_index = [
            sample_indx.index(i) for i in selected_valid + non_valid
        ]
        selected_valid_index = list(set(selected_valid_index))
        selected_valid_index.sort()
        assert len(selected_valid_index) >= self.min_valid_frames

        # read frames and masks
        imgs, labels, boxes, masks, valid = [], [], [], [], []

        mask_dict_path = os.path.join(str(self.img_folder), 'rle_save', video, f'{obj_id}.json')
        with open(mask_dict_path, 'r') as f:
            mask_dict = json.load(f)
        

        for j in range(self.num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            

            mask = np.zeros((h,w))
            if frame_name in mask_dict:
                segm = mask_dict[frame_name]
                if segm is not None:
                    
                    mask = maskUtils.decode(segm)
                

            if (mask > 0).any():
                y1, y2, x1, x2 = self.bounding_box(mask)
                box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:  # some frame didn't contain the instance
                box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)
                # print("{} does not have masks, may cause training noise".format(mask_path))
            mask = torch.from_numpy(mask)
            label = torch.tensor(1)

            # append
            imgs.append(img)
            labels.append(label)
            masks.append(mask)
            boxes.append(box)

        # transform
        # w, h = img.size
        h, w = imgs[0].shape[:2]
        labels = torch.stack(labels, dim=0)
        boxes = torch.stack(boxes, dim=0)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        masks = torch.stack(masks, dim=0)
        target = {
            'frames_idx': torch.tensor(sample_indx),  # [T,]
            'labels': labels,  # [T,]
            'boxes': boxes,  # [T, 4], xyxy
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            "selected_valid": selected_valid_index,
            'video': video,
        }

        return imgs, target

