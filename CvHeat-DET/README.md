<h1 align="center">
  CvHeat-DET
</h1>

<div align="center">

  <h3 align="center">
    Official PyTorch implementation of "Dynamic Graph Induced Contour-aware Heat Conduction Network for Event-based Object Detection"
  </h3>

 <a href="">arXiv</a> &nbsp; 
 <a href="https://github.com/Event-AHU/OpenEvDET/tree/main/CvHeat-DET">GitHub</a>
  
 Xiao Wang<sup>1</sup>, Yu Jin<sup>1</sup>, Wentao Wu<sup>2</sup>, Wei Zhang<sup>3</sup>, Lin Zhu<sup>4</sup>, Bo Jiang<sup>1</sup>, Yonghong Tian<sup>3,5,6</sup>

 <sup>1</sup>School of Computer Science and Technology, Anhui University, Hefei, China
 
 <sup>2</sup>School of Artificial Intelligence, Anhui University, Hefei, China
 
 <sup>3</sup>Peng Cheng Laboratory, Shenzhen, China
 
 <sup>4</sup>Beijing Institute of Technology, Beijing, China
 
 <sup>5</sup>National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University, China
 
 <sup>6</sup>School of Electronic and Computer Engineering, Shenzhen Graduate School, Peking University, China
</div>
---

### Abstract

Event-based Vision Sensors (EVS) have demonstrated significant advantages over traditional RGB frame-based cameras in low-light conditions, high-speed motion capture, and low latency. Consequently, object detection based on EVS has attracted increasing attention from researchers. Current event stream object detection algorithms are typically built upon Convolutional Neural Networks (CNNs) or Transformers, which either capture limited local features using convolutional filters or incur high computational costs due to the utilization of self-attention. Recently proposed vision heat conduction backbone networks have shown a good balance be
tween efficiency and accuracy; however, these models are not specifically designed for event stream data. They exhibit weak capability in modeling object contour information and fail to exploit the benefits of multi-scale features. To address these
 issues, this paper proposes a novel dynamic graph induced contour-aware heat conduction network for event stream based object detection, termed CvHeat-DET. The proposed model effectively leverages the clear contour information inherent in event streams to predict the thermal diffusivity coefficients within the heat conduction model, and integrates hierarchical structural graph features to enhance feature learning across multiple scales. Extensive experiments on three benchmark datasets for event stream-based object detection fully validated the effectiveness of the proposed model.



### Our Proposed Approach

<div align="center">
<img src="https://github.com/Event-AHU/OpenEvDET/blob/5fffa5f2737227535a3c42b96396514d143baaa2/CvHeat-DET/figures/vheat_gnn_framework.jpg" width="800">
</div>



### Experimental Results

<div align="center">
<img src="https://github.com/Event-AHU/OpenEvDET/blob/24d8596f61e434a43b368745d147a5e7ce4fbb1a/CvHeat-DET/figures/det_res.png" width="800">
</div>

---



## Quick start

### Install

We use a single A800 80G GPU for training and evaluation.

```
conda create -n cvheat python=3.8
conda activate cvheat
pip install -r requirements.txt
```

### Data

Download the [EvDET200K (Baiduyun)](https://pan.baidu.com/s/1HfkDyVv_dV_lbJGX0cQEVg?pwd=ahue) dataset or from [[Dropbox](https://www.dropbox.com/scl/fo/2x3qf8bcwd6qb4f70fnda/AL2ULrSzZuVgpVlH8RTqhsY?rlkey=hh7k0lqg1tru4iisi0vo12y6x&st=nz4b3c13&dl=0)], and modify the dataset path in `configs\dataset\EvDET200K_detection.yml`.

### Train

#### training on a single GPU

```
python tools/train.py -c configs/evheat/cvheat.yml
```

#### training on multi-GPU

```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/cvheat.yml -o output/contour
```

### Test

#### testing on a single GPU

```
python tools/train.py -c configs/evheat/cvheat.yml -r path_to_ckp --test-only
```

#### testing on multi-GPU

```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/cvheat.yml -r path_to_ckp --test-only
```

---

## Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate their contributions.

* [vHeat](https://github.com/MzeroMiko/vHeat)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## Citation

If you find this work helps your research, please cite the following paper and give us a star.

```

```
