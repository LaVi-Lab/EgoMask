def refine_bbox_into_5FPS(bbox_info, video_stride=6):
    new_d = []
    mismatch = 0
    bbox_info.sort(key=lambda x: x["frame_number"])
    for d in bbox_info:
        tmp_d = {k:v for k,v in d.items()}
        
        if "exported_clip_frame_number" in d:
            tmp = d["exported_clip_frame_number"]
            if tmp % video_stride > 0:
                new_frame_number = round(tmp / video_stride)
                if tmp_d["frame_number"] != new_frame_number:
                    tmp_d["frame_number"] = new_frame_number
                    mismatch+=1
        else:
            tmp_d["exported_clip_frame_number"] = video_stride * d["frame_number"]
                
        new_d.append(tmp_d)

    print("[MISMATCH]", mismatch)
    
    # remove duplicated
    rm_duplicate_num = len(new_d)
    fnum_map = {d["frame_number"]:d for d in new_d}
    new_d = list(fnum_map.values())
    new_d.sort(key=lambda x: x["frame_number"])
    rm_duplicate_num -= len(new_d)
    print("[RM DUPLICATE]", rm_duplicate_num)
    
    return new_d




def refine_attributes_occurrence(bbox_info, occur_anno):
    numbermap = {d["exported_clip_frame_number"]:d["frame_number"] for d in bbox_info}
    new_occur_anno = []
    not_added = 0
    for d in occur_anno:
        # start
        append_flag = True
        
        if "exported_clip_start_frame_number" in d and "exported_clip_end_frame_number" in d:
            if d["exported_clip_start_frame_number"] not in numbermap:
                print("[exported_clip_start_frame_number]", d["exported_clip_start_frame_number"])
                tmp = d["exported_clip_start_frame_number"]
                if tmp+6 in numbermap:
                    d["start_frame_number"] = numbermap[tmp+6]
                elif tmp-6 in numbermap:
                    d["start_frame_number"] = numbermap[tmp-6]
                else:
                    print("Current_d", d)
                    print("Target number", tmp)
                    print("Nearest number", tmp-6, tmp+6)
                    print("numbermap", numbermap)
                    print("NOT FOUND!")
                    append_flag = False
            else:
                d["start_frame_number"] = numbermap[d['exported_clip_start_frame_number']]
                
            if d["exported_clip_end_frame_number"] not in numbermap:
                print("[exported_clip_end_frame_number]",d["exported_clip_end_frame_number"])
                tmp = d["exported_clip_end_frame_number"]
                if tmp-6 in numbermap:
                    d["end_frame_number"] = numbermap[tmp-6]
                elif tmp+6 in numbermap:
                    d["end_frame_number"] = numbermap[tmp+6]
                else:
                    print("Current_d", d)
                    print("Target number", tmp)
                    print("Nearest number", tmp-6, tmp+6)
                    print("NOT FOUND!")
                    append_flag = False
            else:
                d["end_frame_number"] = numbermap[d['exported_clip_end_frame_number']]
                
        if append_flag:
            new_occur_anno.append(d)
        else:
            not_added += 1
    return new_occur_anno, not_added
        
def build_consistent_appearance(bbox_anno,video_stride=6, relax=2):
    ret_value = []

    start_idx = 0
    end_idx = 0

    for idx, d in enumerate(bbox_anno):
        if idx == 0:
            continue
        # current frame number
        frame_number = d["frame_number"]

        # if continue
        if frame_number - bbox_anno[idx - 1]["frame_number"] <= relax:
            end_idx = idx
        else:
            # if not continue
            tmp = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_frame_number": bbox_anno[start_idx]["frame_number"],
                "end_frame_number": bbox_anno[end_idx]["frame_number"],
                "video_start_frame_number": bbox_anno[start_idx]["video_frame_number"] if "video_frame_number" in bbox_anno[start_idx] else None,
                "exported_clip_start_frame_number": bbox_anno[start_idx][
                    "exported_clip_frame_number"
                ] if "exported_clip_frame_number" in bbox_anno[start_idx] else video_stride * bbox_anno[start_idx]["frame_number"],
                "video_end_frame_number": bbox_anno[end_idx]["video_frame_number"] if "video_frame_number" in bbox_anno[end_idx] else None,
                "exported_clip_end_frame_number": bbox_anno[end_idx][
                    "exported_clip_frame_number"
                ] if "exported_clip_frame_number" in bbox_anno[end_idx] else video_stride * bbox_anno[end_idx]["frame_number"],
            }
            ret_value.append(tmp)

            start_idx = idx
            end_idx = start_idx

    if start_idx != end_idx:
        tmp = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_frame_number": bbox_anno[start_idx]["frame_number"],
            "end_frame_number": bbox_anno[end_idx]["frame_number"],
            "video_start_frame_number": bbox_anno[start_idx]["video_frame_number"] if "video_frame_number" in bbox_anno[start_idx] else None,
            "exported_clip_start_frame_number": bbox_anno[start_idx][
                "exported_clip_frame_number"
            ] if "exported_clip_frame_number" in bbox_anno[start_idx] else video_stride * bbox_anno[start_idx]["frame_number"],
            "video_end_frame_number": bbox_anno[end_idx]["video_frame_number"] if "video_frame_number" in bbox_anno[end_idx] else None,
            "exported_clip_end_frame_number": bbox_anno[end_idx][
                "exported_clip_frame_number"
            ] if "exported_clip_frame_number" in bbox_anno[end_idx] else video_stride * bbox_anno[end_idx]["frame_number"],
        }
    ret_value.append(tmp)
    return ret_value

def get_clip_narrations(clip_uid, clip2video_uid_data, narration_data):
    video_d = clip2video_uid_data[clip_uid]
    video_narration = narration_data[video_d["video_uid"]]
    video_start_frame, video_end_frame = (
        video_d["video_start_frame"],
        video_d["video_end_frame"],
    )
    if video_narration["status"]!="complete":
        print(clip_uid, video_narration["status"])
    # assert video_narration["status"] == "complete"

    def deal_with_each_narration_annotation(anno,video_stride=6):
        narrations = anno["narrations"]
        summaries = anno["summaries"]

        new_narrations = []
        for nar in narrations:
            if nar["timestamp_frame"] < video_start_frame:
                continue
            elif video_start_frame <= nar["timestamp_frame"] <= video_end_frame:
                nar["clip_timestamp_frame"] = nar["timestamp_frame"] - video_start_frame
                nar["estimated_5FPS_frame_number"] =  round(nar["clip_timestamp_frame"] / video_stride)
                new_narrations.append(nar)
            elif nar["timestamp_frame"] > video_end_frame:
                break

        if len(new_narrations) == 0:
            return None
        
        timestamp_range = (
            new_narrations[0]["timestamp_sec"],
            new_narrations[-1]["timestamp_sec"],
        )

        new_summaries = []
        for summ in summaries:
            if summ["end_sec"] < timestamp_range[0]:
                continue
            elif summ["start_sec"] > timestamp_range[-1]:
                break
            else:
                new_summaries.append(summ)
        return dict(narrations=new_narrations, summaries=new_summaries)

    clip_narration_d = {}
    for key, value in video_narration.items():
        if key != "status":
            clip_narration_d[key] = deal_with_each_narration_annotation(value)
    clip_narration_d["status"] = video_narration["status"]
    return clip_narration_d
