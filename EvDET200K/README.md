<h1 align="center">
  MvHeat-DET
</h1>

<div align="center">

  <h3 align="center">
    Official PyTorch implementation of "Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset"
  </h3>
  
  <a href="https://arxiv.org/abs/2412.06647">arXiv</a> &nbsp; 
  <a href="https://github.com/Event-AHU/OpenEvDET/edit/main/EvDET200K">GitHub</a>
  
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
Object detection in event streams has emerged as a cutting edge research area, demonstrating superior performance in low-light conditions, scenarios with motion blur, and rapid movements. Current detectors leverage spiking neural networks, Transformers, or convolutional neural networks as their core architectures, each with its own set of limitations including restricted performance, high computational overhead, or limited local receptive fields. This paper introduces a novel MoE (Mixture of Experts) heat conduction based object detection algorithm that strikingly balances accuracy and computational efficiency. Initially, we employ a stem network for event data embedding, followed by processing through our innovative MoE-HCO blocks. Each block integrates various expert modules to mimic heat conduction within event streams. Subsequently, an IoU-based query selection module is utilized for efficient token extraction, which is then channeled into a detection head for the final object detection process. Furthermore, we are pleased to introduce EvDET200K, a novel benchmark dataset for event-based object detection. Captured
 with a high-definition Prophesee EVK4-HD event camera, this dataset encompasses 10 distinct categories, 200,000 bounding boxes, and 10,054 samples, each spanning 2 to 5 seconds. We also provide comprehensive results from over 15 state-of-the-art detectors, offering a solid foundation for future research and comparison. 



### Our Proposed Approach 
<div align="center">
<img src="https://github.com/Event-AHU/OpenEvDET/blob/1fd7f11fa87d8c70a19986cdc36613c855f4fe32/EvDET200K/figures/framework.png" width="800">
</div>

### Experimental Results
<div align="center">
<img src="https://github.com/Event-AHU/OpenEvDET/blob/1fd7f11fa87d8c70a19986cdc36613c855f4fe32/EvDET200K/figures/benchmarkResults.png" width="800">
</div>

### Dataset visualizations
<div align="center">
<img src="https://github.com/Event-AHU/OpenEvDET/blob/58d0c56314cb7981a973885370ba192a5b0ed611/EvDET200K/figures/dataset_visualization.jpg" width="800">
<!-- <img src="https://github.com/Event-AHU/OpenEvDET/blob/1fd7f11fa87d8c70a19986cdc36613c855f4fe32/EvDET200K/figures/benchmark_dataset_compare.png" width="800"> -->
</div>

---
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


### Train

#### training on single-gpu
```
CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/evheat/MvHeatDET.yml
```
#### training on multi-gpu
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml
```

### Test

#### testing on single-gpu
```
python tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```
#### testing on multi-gpu
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py -c configs/evheat/MvHeatDET.yml -r ckp/mvheatdet_input640_layers18_dim768.pth --test-only
```

---
## Checkpoint Download
You can download the pretrained checkpoint on EvDET200K dataset from [BaiduYun](https://pan.baidu.com/s/1UZ3TH0VA31XvVaH-BpiTKA?pwd=ahue), with model config:

<table>
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Input Size</th>
    <th align="center">Block Num.</th>
    <th align="center">Channel</th>
  </tr>
  <tr>
    <td align="center">EvDET200K</td>
    <td align="center">640</td>
    <td align="center">(2,2,18,2)</td>
    <td align="center">(96,192,384,768)</td>
  </tr>
</table>

---

## Acknowledgments

Our code is extended from the following repositories. We sincerely appreciate for their contributions.

* [vHeat](https://github.com/MzeroMiko/vHeat)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

## Cite
If you find this work helps your research, please cite the following paper and give us a star.
```
@article{wang2024mvheatdet,
  title={Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset},
  author={Wang, Xiao and Jin, Yu and Wu, Wentao and Zhang, Wei and Zhu, Lin and Jiang, Bo and Tian, Yonghong},
  journal={arXiv preprint arXiv:2412.06647},
  year={2024}
}
```
