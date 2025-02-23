



### Abstract 
Object detection in event streams has emerged as a cutting-edge research area, demonstrating superior performance in low-light conditions, scenarios with motion blur, and rapid movements. Current detectors leverage spiking neural networks, Transformers, or convolutional neural networks as their core architectures, each with its own set of limitations including restricted performance, high computational overhead, or limited local receptive fields. This paper introduces a novel MoE (Mixture of Experts) heat conduction-based object detection algorithm that strikingly balances accuracy and computational efficiency. Initially, we employ a stem network for event data embedding, followed by processing through our innovative MoE-HCO blocks. Each block integrates various expert modules to mimic heat conduction within event streams. Subsequently, an IoU-based query selection module is utilized for efficient token extraction, which is then channeled into a detection head for the final object detection process. Furthermore, we are pleased to introduce EvDET200K, a novel benchmark dataset for event-based object detection. Captured with a high-definition Prophesee EVK4-HD event camera, this dataset encompasses 10 distinct categories, 200,000 bounding boxes, and 10,054 samples, each spanning 2 to 5 seconds. We also provide comprehensive results from over 15 state-of-the-art detectors, offering a solid foundation for future research and comparison. 



### Framework 
<p align="center">
  <img src="https://github.com/Event-AHU/OpenEvDET/blob/main/EvDET200K/figures/framework.png" alt="benchmarkResults.png" width="800"/>
</p>


### Visualization of Dataset 
<p align="center">
  <img src="https://github.com/Event-AHU/OpenEvDET/blob/main/EvDET200K/figures/dataset_visualization.jpg" alt="dataset_visualization.jpg" width="800"/>  
</p>



### Comparison of Existing Event-based Benchmark Dataset  
<p align="center">
  <img src="https://github.com/Event-AHU/OpenEvDET/blob/main/EvDET200K/figures/benchmark_dataset_compare.png" width="800"/>
</p>


### Benchmark Results 
<p align="center">
  <img src="https://github.com/Event-AHU/OpenEvDET/blob/main/EvDET200K/figures/benchmarkResults.png" alt="benchmarkResults.png" width="800"/>  
</p>



