# HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers (CVPR21 Oral)

## Environment
Require Python3, CUDA>=10.1, and torch>=1.4, all dependencies are as follows:
```shell script
pip3 install torch==1.4.0 torchvision==0.5.0 opencv-python tqdm tensorboard lmdb pyyaml packaging Pillow==6.2.2 matplotlib yacs pyarrow==0.17.1
pip3 install cityscapesscripts  # for Cityscapes segmentation
pip3 install mmcv-full==latest+torch1.4.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html  # for Segmentation data loader
pip3 install pycocotools shapely==1.6.4 Cython pandas pyyaml json_tricks scikit-image  # for COCO keypoint estimation
```
or ```pip3 install requirements.txt```

## Setup

Optionally configure NCCL before running:
```shell script
export NCCL_IB_DISABLE=1 
export NCCL_IB_HCA=mlx5_0 
export NCCL_IB_GID_INDEX=3 
export NCCL_SOCKET_IFNAME=eth0 
export HOROVOD_MPI_THREADS_DISABLE=1
export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,compact,1,0
```
Set the following ENV variable:
```
$MASTER_ADDR: IP address of the node 0 (Not required if you have only one node (machine))
$MASTER_PORT: Port used for initializing distributed environment
$NODE_RANK: Index of the node
$N_NODES: Number of nodes 
$NPROC_PER_NODE: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES`)
```

Example1 (One machine with 8 GPUs):
```
Node 1: 
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=1 --node_rank=0 --master_port=1234 train.py
```

Example2 (Two machines, each has 8 GPUs):
```
Node 1: (IP: 192.168.1.1, and has a free port: 1234)
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
--master_port=1234 train.py

Node 2:
>>> python -m torch.distributed.launch --nproc_per_node=8
--nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
--master_port=1234 train.py
```

## Datasets

1. ImageNet
    - Prepare ImageNet data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
    - Optional: Generate lmdb dataset by `utils/lmdb_dataset.py`. If not, please overwrite `dataset:imagenet1k_lmdb` in yaml to `dataset:imagenet1k`.
    - The directory structure of `$DATA_ROOT` should look like this:
    ```
    ${DATA_ROOT}
    ├── imagenet
    └── imagenet_lmdb
    ```
    - Link the data:
    ```shell script
    ln -s YOUR_LMDB_DIR data/imagenet_lmdb
    ```
   
2. Cityscapes
    - Download data from [Cityscapes](https://www.cityscapes-dataset.com/).
    - unzip gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/cityscapes
    ```
    - preprocess the data:
    ```shell script
    python3 tools/convert_cityscapes.py data/cityscapes --nproc 8
    ```
      
3. ADE20K
    - Download data from [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
    - unzip ADEChallengeData2016.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/ade20k
    ```

4. COCO keypoints
    - Download data from [COCO](https://cocodataset.org/#download).
    - build tools
    ```shell script
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python3 setup.py build_ext --inplace
    python3 setup.py build_ext install
    make  # for nms
    ```
    - Unzip and Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/coco
    ```
    - We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
    - Download and extract them under ```data/coco/person_detection_results```, and make them look like this:
    ```
    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            |-- person_detection_results
            |   |-- COCO_val2017_detections_AP_H_56_person.json
            |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
            `-- images
                |-- train2017
                |   |-- 000000000009.jpg
                |   |-- 000000000025.jpg
                |   |-- 000000000030.jpg
                |   |-- ... 
                `-- val2017
                    |-- 000000000139.jpg
                    |-- 000000000285.jpg
                    |-- 000000000632.jpg
                    |-- ... 
    ```

## Running (train & evaluation)
- Search for NAS models.
    ```shell script
    python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --nnodes=${N_NODES} \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        --use_env train.py app:configs/YOUR_TASK.yml
    ```
    Supported tasks:
    - cls_imagenet
    - seg_cityscapes
    - seg_ade20k
    - keypoint_coco
    
    The super network is constructed using ```model_kwparams``` in YOUR_TASK.yml.  
    To enable the searching of Transformer, set ```prune_params.use_transformer=True``` in YOUR_TASK.yml,
  the token numbers of each Transformer will be printed during training.  
    The searched architecture can be found in ```best_model.json``` in the output dir.  
  

- Retrain the searched models.
    - For retraining the searched classification model, please use ```best_model.json``` to overwrite the ```checkpoint.json``` in root dir of this project.
    
    - Modify ```models/hrnet.py``` to set ```checkpoint_kwparams = json.load(open('checkpoint.json'))``` and 
    ```class InvertedResidual(InvertedResidualChannelsFused)```
    
    - Retrain the model.
    ```shell script
    python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --nnodes=${N_NODES} \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        --use_env train.py app:configs/cls_retrain.yml
    ```

## Miscellaneous
1. Plot keypoint detection results.
    ```shell script
    python3 tools/plot_coco.py --prediction output/results/keypoints_val2017_results_0.json --save-path output/vis/
    ```

2. About YAML config
- The codebase is a general ImageNet training framework using yaml config with several extension under `apps` dir, based on PyTorch.
    - YAML config with additional features
        - `${ENV}` in yaml config.
        - `_include` for hierachy config.
        - `_default` key for overwriting.
        - `xxx.yyy.zzz` for partial overwriting.
    - `--{{opt}} {{new_val}}` for command line overwriting.

3. Any questions regarding HR-NAS, feel free to contact the author (mingyuding@hku.hk).
4. If you find our work useful in your research please consider citing our paper:
    ```
    @inproceedings{ding2021hrnas,
      title={HR-NAS: Searching Efficient High-Resolution Neural Architectures with Lightweight Transformers},
      author={Ding, Mingyu and Lian, Xiaochen and Yang, Linjie and Wang, Peng and Jin, Xiaojie and Lu, Zhiwu and Luo, Ping},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2021}
    }
    ```
