import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
sys.path.append(os.getcwd())
import time
import argparse
import numpy as np
import pycocotools.mask as maskUtils
from evaluation.metrics import db_eval_iou, db_eval_boundary, db_eval_boundary_temporal
import multiprocessing as mp
from evaluation.common_utils import *


NUM_WOEKERS = 32


def eval_queue(q, rank, out_dict, mask_dir, pred_path):
    while not q.empty():
        vid_name, exp_id, obj_id, vid_type = q.get()
        
        vid = meta_exp[vid_name]
        exp_name = f"{vid_name}_{exp_id}_{obj_id}"
        T_dict = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "gt_temporal": None,
            "res_temporal": None,
        }
        if not os.path.exists(f"{pred_path}/{vid_name}"):
            print(f"Prediction for {vid_name} not found, skipping...")
            continue
        
        raw_clip_name = vid_name.split("--")[0] if vid_type == "medium" else vid_name
        
        # Load ground truth masks
        gold_mask_rle = read_json(
            os.path.join(mask_dir, f"{raw_clip_name}/{obj_id}.json")
        )
        
        # Load predicted masks
        pred_mask_rle = read_json(
            os.path.join(
                pred_path, f"{vid_name}/{exp_id}/{exp_id}-{obj_id}.json"
            )
        )
        
        # detection info for SAM2 grounding...
        grounding_ret_info = None
        grounding_ret = os.path.join(pred_path,f"{vid_name}/{exp_id}/{exp_id}-{obj_id}_grounding_ret.json")
        if os.path.exists(grounding_ret):
            grounding_ret_info = read_json(grounding_ret)
            
        
        # get h,w
        for fname, mask_rle in gold_mask_rle.items():
            mask = maskUtils.decode(mask_rle)
            if mask is not None:
                h, w = mask.shape
                break
        
        vid_len = len(vid["frames"])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        
        # for IoU_all
        overall_iou_list = []
        # for IoU_gold
        gold_iou_list = []
        # for IoU_gold_with_pred
        gold_with_pred_iou_list = []
        
        detection_flag = False
        detection_bbox_iou = 0.0
        detection_mask_iou = 0.0
        
        for fidx, fname in enumerate(vid["frames"]):
            golden_flag = False
            
            if fname in gold_mask_rle:
                gold_segm = gold_mask_rle[fname]
                if gold_segm is not None:
                    golden_flag = True
                    mask = maskUtils.decode(gold_segm)
                    gt_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)
            if fname in pred_mask_rle:
                pred_segm = pred_mask_rle[fname]
                if pred_segm is not None:
                    mask = maskUtils.decode(pred_segm)
                    pred_masks[fidx, :, :] = np.array(mask, dtype=np.uint8)

            overlap = np.logical_and(gt_masks[fidx], pred_masks[fidx])
            union = np.logical_or(gt_masks[fidx], pred_masks[fidx])
            
            
            if union.sum() > 0:
                iou_ = overlap.sum() / union.sum()
                overall_iou_list.append(iou_)
            else: 
                # no gt mask & no pred mask
                overall_iou_list.append(1.0)

            if fname in gold_mask_rle:
                gold_iou_list.append(iou_)
                gold_with_pred_iou_list.append(iou_)
            elif fname in pred_mask_rle:
                # only pred mask, iou_ = 0.0
                # penalize this case
                gold_with_pred_iou_list.append(iou_)
            
            if grounding_ret_info is not None and fname == grounding_ret_info["frame_name"]:
                
                detection_mask_iou = iou_
                # if golden
                if golden_flag:
                    detection_flag = True
                    # bbox_iou
                    pred_bbox = grounding_ret_info["input_boxes"]
                    gold_bbox = mask2bbox(gt_masks[fidx])
                    detection_bbox_iou = calculate_iou(pred_bbox, gold_bbox)

                

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        t = db_eval_boundary_temporal(gt_masks, pred_masks)

        overall_iou = np.mean(overall_iou_list)
        gold_iou = np.mean(gold_iou_list)
        gold_with_pred_iou = np.mean(gold_with_pred_iou_list)
        out_dict[exp_name] = {
            "J": j,
            "F": f,
            "T": t,
            "overall_iou": overall_iou,
            "gold_iou": gold_iou,
            "gold_with_pred_iou": gold_with_pred_iou,
            "detection_flag": detection_flag,
            "detection_bbox_iou": detection_bbox_iou,
            "detection_mask_iou": detection_mask_iou,
        }
        
if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="long", choices=["long", "medium", "short", "full"])
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the predictions directory")
    parser.add_argument("--save_name", default="exp_results.json", type=str)
    args = parser.parse_args()
    args.pred_path = os.path.join(args.pred_path, args.dataset_type)
    
    
    # start
    queue = mp.Queue()
    output_dict = mp.Manager().dict()

    # load dataset
    dataset_info = dataset_mapping[args.dataset_type]
    
    meta_exp_path = dataset_info["meta_expression_file"]
    meta_exp = read_json(meta_exp_path)["videos"]
    shared_meta_exp = mp.Manager().dict(meta_exp)
    output_dict = mp.Manager().dict()
    
    for vid, vid_info in meta_exp.items():
        if args.dataset_type == "full":
            vid_type = vid_info["subset"]
        else:
            vid_type = args.dataset_type
        for exp_id, exp_info in vid_info["expressions"].items():
            obj_id = exp_info["obj_id"]
            queue.put([vid, str(exp_id), obj_id,vid_type])
    
    cnt = queue.qsize()
    print("Q-Size:", queue.qsize())
    
    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(
            target=eval_queue,
            args=(
                queue,
                rank,
                output_dict,
                dataset_info["mask_dir"],
                args.pred_path,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print(len(output_dict))
    assert cnt == len(output_dict)
    
    j = [output_dict[x]["J"] for x in output_dict]
    f = [output_dict[x]["F"] for x in output_dict]
    t_f1 = [output_dict[x]["T"]["f1"] for x in output_dict]
    t_acc = [output_dict[x]["T"]["accuracy"] for x in output_dict]
    t_precision = [output_dict[x]["T"]["precision"] for x in output_dict]
    t_recall = [output_dict[x]["T"]["recall"] for x in output_dict]

    iou_overall = [output_dict[x]["overall_iou"] for x in output_dict]
    iou_gold = [output_dict[x]["gold_iou"] for x in output_dict]
    iou_gold_with_pred = [output_dict[x]["gold_with_pred_iou"] for x in output_dict]
    
    detection_flag = [1 if output_dict[x]["detection_flag"] else 0 for x in output_dict]
    detection_bbox_iou = [output_dict[x]["detection_bbox_iou"] for x in output_dict]
    detection_mask_iou = [output_dict[x]["detection_mask_iou"] for x in output_dict]
    
    
    output_path = os.path.join(args.pred_path, "default_" + args.save_name)
    
    
    results = {
        "J": round(100 * float(np.mean(j)), 2),
        "F": round(100 * float(np.mean(f)), 2),
        "J&F": round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
        "T_f1": round(100 * float(np.mean(t_f1)), 2),
        "T_acc": round(100 * float(np.mean(t_acc)), 2),
        "T_precision": round(100 * float(np.mean(t_precision)), 2),
        "T_recall": round(100 * float(np.mean(t_recall)), 2),
        "iou_overall": round(100 * float(np.mean(iou_overall)), 2),
        "iou_gold": round(100 * float(np.mean(iou_gold)), 2),
        "iou_gold_with_pred": round(100 * float(np.mean(iou_gold_with_pred)), 2),
        "detection_flag": round(100 * float(np.mean(detection_flag)), 2),
        "detection_bbox_iou": round(100 * float(np.mean(detection_bbox_iou)), 2),
        "detection_mask_iou": round(100 * float(np.mean(detection_mask_iou)), 2),
    }
    print("[INFO] Writing to", output_path)
    write_json(results, output_path)
    
    
    output_jsonl = [dict(seq_exp=k, **v) for k, v in output_dict.items()]
    write_jsonl(output_jsonl, output_path.replace(".json", "_full_result.jsonl"))
    
    total_time =  time.time() - start_time
    print("time: %.4f s" % (total_time))