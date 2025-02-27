##MvHeat-DET

<div align="center">
 
 <img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/promptpar_logo.png" width="600">


 **Official PyTorch implementation of "Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset"**

 ------
 
</div>

> **[Official PyTorch implementation of "Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset]()**, Xiao Wang, Yu Jin, Wentao Wu, Wei Zhang, Lin Zhu, Bo Jiang, Yonghong Tian


## Quick start
### Install
we use single RTX 4090 24G GPU for training and evaluation. 
```
conda create -n mvheat python=3.8
conda activate mvheat
pip install -r requirements.txt
```

### Data
Download the EvDET200K dataset, and modify the dataset path in `configs\dataset\EvDET200K_detection.yml`.


## Train

### training on single-gpu
```
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/evheat/MvHeatDET.yml
```
### training on multi-gpu
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml
```

## Test

### testing on single-gpu
```
python tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```
### testing on multi-gpu
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```

## Checkpoint Download
You can download the pretrained checkpoint from [BaiduYun]().
## News: 


## Abstract 
Object detection in event streams has emerged as a cutting edge research area, demonstrating superior performance in low-light conditions, scenarios with motion blur, and rapid movements. Current detectors leverage spiking neural networks, Transformers, or convolutional neural networks as their core architectures, each with its own set of limitations including restricted performance, high computational overhead, or limited local receptive fields. This paper introduces a novel MoE (Mixture of Experts) heat conduction based object detection algorithm that strikingly balances accuracy and computational efficiency. Initially, we employ a stem network for event data embedding, followed by processing through our innovative MoE-HCO blocks. Each block integrates various expert modules to mimic heat conduction within event streams. Subsequently, an IoU-based query selection module is utilized for efficient token extraction, which is then channeled into a detection head for the final object detection process. Furthermore, we are pleased to introduce EvDET200K, a novel benchmark dataset for event-based object detection. Captured
 with a high-definition Prophesee EVK4-HD event camera, this dataset encompasses 10 distinct categories, 200,000 bounding boxes, and 10,054 samples, each spanning 2 to 5 seconds. We also provide comprehensive results from over 15 state-of-the-art detectors, offering a solid foundation for future research and comparison. 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/frontImage.jpg" width="800">





## Our Proposed Approach 
<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/pipeline.jpg" width="800">




## Experimental Results 

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/featuremap_vis.png" width="800">

<img src="https://github.com/Event-AHU/OpenPAR/blob/main/PromptPAR/figures/attResults_vis.jpg" width="800">

### Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate for their contributions.

* [vHeat](https://github.com/MzeroMiko/vHeat)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
