# Structure

The structure of `EgoMask` (Benchmark) is as follows: 
```
├─ egomask
    ├─ JPEGImages
    │   ├─ refego
    │   └─ egotracks
    ├─ annotations
    ├─ subset
    │   ├─ long
    │   │   ├─ meta_expressions.json
    │   │   └─ meta.json
    │   ├─ medium
    │   │   └─ meta_expressions.json
    │   └─ short
    │       ├─ meta_expressions.json
    │       └─ meta.json
    ├─ meta_expressions.json
    └─ meta.json

```
The structure of `EgoMask-Train` (Training dataset) is as follows: 
```
├─ egomask-train
    ├─ meta_expressions
    │   └─ train
    │       └─ meta_expressions.json 
    └─ train
        ├─ JPEGImages
        ├─ mask_dict.json
        └─ meta.json
```

# Dataset Prepare

## Download Annotations


We have provided the annotations in [Hugging Face](https://huggingface.co/datasets/XuuuXYZ/EgoMask).

```bash
hf download XuuuXYZ/EgoMask --repo-type dataset # --local-dir [PATH_TO_EGOMASK_DATA]
```


## Process JPEGImages

Please follow [Start here](https://github.com/EGO4D/docs/blob/main/docs/start-here.md) for instructions on how to access Ego4D dataset. Then use the following scripts to process images. Note that our aws profile name is set to `ego4d`. 


For EgoMask (Benchmark)

```bash
bash preprocess/process_refego.sh
bash preprocess/process_egotracks_for_benchmark.sh
```

For EgoMask-Train (Training data)
```bash
bash preprocess/process_egotracks_fro_training.sh
```