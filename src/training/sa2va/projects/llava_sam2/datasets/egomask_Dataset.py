import json
from tqdm import tqdm
from .ReVOS_Dataset import VideoReVOSDataset



def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def safe_np_random_choice(l, select_k,replace=True):
    if len(l) == 0 or select_k <= 0:
        return []
    elif len(l) >= select_k:
        selected = np.random.choice(l, select_k, replace=replace)
    else:
        remaining_ = select_k - len(l)
        selected_others = np.random.choice(l, remaining_, replace=True)
        selected = np.array(l+selected_others.tolist())
    return selected
    

class VideoEgoMaskDataset(VideoReVOSDataset):
    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        len_frames = len(data_dict[0]['frames'])
        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])

        # prepare images, random select k frames
        
        valid_frames = set(data_dict[0]['valid_frames'])
        
        valid_frame_idx_list = []
        negative_frame_idx_list = []
        for idx, fname in  enumerate(data_dict[0]["frames"]):
            if fname in valid_frames:
                valid_frame_idx_list.append(idx)
            else:
                negative_frame_idx_list.append(idx)

        assert len(valid_frame_idx_list) > 0, f'valid_frame_idx_list is empty, len_frames: {len_frames}, valid_frames: {valid_frames}, data_dict[0]["frames"]: {data_dict[0]["frames"]}, data_dict[0]["valid_frames"]: {data_dict[0]["valid_frames"]}'
        continuous_can_selected = []
        for valid_frame_idx in valid_frame_idx_list:
            start_ = max(0, valid_frame_idx -  select_k + 1)
            end_ = min(valid_frame_idx + select_k, len_frames - select_k)
            continuous_can_selected.extend(list(range(start_, end_)))
        continuous_can_selected = list(set(continuous_can_selected))
        continuous_can_selected.sort()
        
        if len_frames > select_k + 1:
            if self.frame_contiguous_sample and random.random() < 0.5:
                
                selected_start_frame = np.random.choice(continuous_can_selected, 1, replace=False)
                selected_frame_indexes = [selected_start_frame[0] + _i for _i in range(select_k)]
            else:
                golden_frame_num = random.choice(list(range(1,select_k)))
                
                selected_golden = safe_np_random_choice(valid_frame_idx_list, golden_frame_num, replace=False).tolist()
                selected_negative = safe_np_random_choice(negative_frame_idx_list, select_k - golden_frame_num, replace=False).tolist()
                
                selected_frame_indexes = np.array(selected_golden+selected_negative)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        
        # make sure the selected frames contains valid frames
        assert set(selected_frame_indexes.tolist()) & set(valid_frame_idx_list) != set()
        
        selected_frame_indexes.sort()

        if self.use_fast:
            # sample fast branch
            fast_interval = len_frames / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len_frames - 1) for i in range(self.n_fast_images)]
            fast_video_frames = []
            for selected_frame_index in sampled_fast_frame_idxs:
                frame_id = data_dict[0]['frames'][selected_frame_index]
                fast_video_frames.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))
        else:
            fast_video_frames = None
            sampled_fast_frame_idxs = None

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))
        
        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        if self.use_fast:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token,
                                          n_fast_images=len(fast_video_frames),)
        else:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token)


        # prepare masks
        video_masks = []
        for object_info in data_dict:
            anno_ids = object_info['mask_anno_id']

            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = self.mask_dict[anno_id]
                frames_masks_ = []
                for frame_idx in selected_frame_indexes:
                    frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        if self.use_fast:
            fast_video_masks = []
            assert sampled_fast_frame_idxs is not None
            for object_info in data_dict:
                anno_ids = object_info['mask_anno_id']
                obj_masks = []
                for anno_id in anno_ids:
                    anno_id = str(anno_id)
                    frames_masks = self.mask_dict[anno_id]
                    frames_masks_ = []
                    for frame_idx in sampled_fast_frame_idxs:
                        frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                    obj_masks.append(frames_masks_)
                fast_video_masks.append(obj_masks)
        else:
            fast_video_masks = None

        ret = {
            'images': images, 
            'video_masks': video_masks, 
            'conversation': text_dict['conversation'],
            'fast_images': fast_video_frames, 
            'fast_video_masks': fast_video_masks
        }
        
        return ret
    
    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        mask_dict = read_json(mask_file)
        expression_datas = read_json(expression_file)["videos"]

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in tqdm(expression_datas):
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                
                
                
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                
                obj_id = exp_dict['obj_id']
                vid_meta_key = str((vid_name, obj_id))
                meta['mask_anno_id'] = [vid_meta_key]

                meta['obj_id'] = [obj_id]

                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                
                current_mask = mask_dict[vid_meta_key]
                valid_index = [idx for idx, v in enumerate(current_mask) if v is not None]

                if len(valid_index) == 0:
                    print(f'[INFO] {vid_meta_key} has no valid mask')
                    continue
                meta["valid_frames"] = [vid_frames[i] for i in valid_index]
                
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                
                
                if vid_meta_key not in vid2metaid.keys():
                    vid2metaid[vid_meta_key] = []
                vid2metaid[vid_meta_key].append(len(metas) - 1)
        
        print('[INFO] (len(vid2metaid), len(metas), len(mask_dict)) = ', (len(vid2metaid), len(metas), len(mask_dict)))
        
        return vid2metaid, metas, mask_dict