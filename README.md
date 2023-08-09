# MobileViG
MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications [PDF](https://openaccess.thecvf.com/content/CVPR2023W/MobileAI/papers/Munir_MobileViG_Graph-Based_Sparse_Attention_for_Mobile_Vision_Applications_CVPRW_2023_paper.pdf)

Mustafa Munir and William Avery

# Overview
This repository contains the source code for MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications


# Pretrained Models

Weights trained on ImageNet-1K can be downloaded [here](https://huggingface.co/SLDGroup/MobileViG/tree/main). 

Weights trained on COCO 2017 Object Detection and Instance Segmentation can be downloaded [here](https://huggingface.co/SLDGroup/MobileViG/tree/main/Detection). 

### detection
Contains all of the object detection and instance segmentation results, backbone code, and config.

### models
Contains the main MobileViG model code.

### util
Contains utility scripts used in MobileViG.

# Usage

## Installation Image Classification

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
conda install mpi4py
```
```
pip install -r requirements.txt
```
## Image Classification

### Train image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model mobilevig_model --output_dir mobilevig_results
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path ../../Datasets/ILSVRC/Data/CLS-LOC/ --model mobilevig_m --output_dir mobilevig_test_results
```
### Test image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model mobilevig_model --resume pretrained_model --eval
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path ../../Datasets/ILSVRC/Data/CLS-LOC/ --model mobilevig_s --resume Pretrained_Models_MobileViG/MobileViG_S_78_2.pth.tar --eval
```


## Installation Object Detection and Instance Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install timm
```
```
pip install submitit
```
```
pip install -U openmim
```
```
mim install mmcv-full
```
```
mim install mmdet==2.28
```
## Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

All commands for object detection and instance segmentation should be run from the MobileViG/detection/ directory.

### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).

### ImageNet Pretraining
Put ImageNet-1K pretrained weights of backbone as 
```
MobileViG
├── Final_Results
│   ├── model
│   │   ├── model.pth.tar
│   │   ├── ...
```

### Train object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node num_GPUs --nnodes=num_nodes --node_rank 0 main.py configs/mask_rcnn_mobilevig_model --mobilevig_model mobilevig_model --work-dir Output_Directory --launcher pytorch > Output_Directory/log_file.txt 
```
For example:
```
python -m torch.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 main.py configs/mask_rcnn_mobilevig_m_fpn_1x_coco.py --mobilevig_model mobilevig_m --work-dir detection_results/ --launcher pytorch > detection_results/mobilevig_m_run_test.txt 
```
### Test object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --node_rank 0 test.py configs/mask_rcnn_mobilevig_model --checkpoint Pretrained_Model --eval {bbox or segm} --work-dir Output_Directory --launcher pytorch > log_file.txt
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 test.py configs/mask_rcnn_mobilevig_m_fpn_1x_coco.py --checkpoint ../Pretrained_Models_MobileViG/Detection/det_mobilevig_m_62_8.pth --eval bbox --work-dir detection_results/ --launcher pytorch > detection_results/mobilevig_m_run_evaluation.txt
```


### Citation
```
@InProceedings{Munir_2023_CVPR,
    author    = {Munir, Mustafa and Avery, William and Marculescu, Radu},
    title     = {MobileViG: Graph-Based Sparse Attention for Mobile Vision Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2210-2218}
}
```

