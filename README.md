# Long-term Spatio-temporal Network  
This is the implementation of Long-term Spatio-temporal Network for Video Recognition  
Now in experimental release, suggestions welcome.
## Paper  
Zhenwei Wang, Wei Dong, Bingbing Zhang, and Jianxin Zhang. LSN: Long-term Spatio-temporal Network for Video Recognition.
## Introduction  
Though Long short-term memory (LSTM) is widely exploited to process temporal or sequential data, it has attracted too little attention in the current mainstream field of video action recognition. Therefore, this work attempts to model long-term spatio-temporal information of video action based on LSTM, and proposes a novel Long-term Spatio-temporal Network (LSN). The core of LSN is composed of the high-order ConvLSTM (HO-ConvLSTM) module and 2D convolution. More specifically, we adopt HO-ConvLSTM module that consists of an Accumulated Temporary State (ATS) module and a standard ConvLSTM as a bottleneck structure with residual connections, and then insert HO-ConvLSTM into different stages of 2D convolutional neural networks (CNNs). In ATS module, several previous hidden states are combined to one temporary state, and the update of the current hidden state is up to the fusion one. By doing so, the longer temporal correlation of the spatio-temporal features are captured in different spatial resolutions. The experiments on three commonly used video benchmarks demonstrate that our method can achieve performance matching or better than the state-of-the-arts.
## Framework  
![](https://github.com/weiwei-max/LSN/blob/main/LSN_framework.jpg)  
* Figure 1 shows the framework of Long-term Spatio-temporal Network
## Environment      
* Linux OS     
* Pytorch 1.6+    
* CUDA 10.0+    
* CUDNN 7.0+
## Installation and Usage  
* Always use `git clone --recursive git@https://github.com/weiwei-max/LSN.git` to clone this project.  
* Download the video datasets ([Something-Something V1](https://20bn.com/datasets/something-something/v1), [Something-Something V2](https://20bn.com/datasets/something-something/v2), [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html)).
* Extract RGB frames from Video (More details refer to [TSN](https://github.com/yjxiong/temporal-segment-networks)).  
* Run the scripts to start the experiments. 
## Results  
Methods | frames | frames x clips x crops | S-S V1 (Top-1 Acc, %) | S-S V2 (Top-1 Acc, %)  | Diving48 (Top-1 Acc, %)  | 
:----:    | :----:     |:----------:         |:-------:   | :--------:  | :-----: | 
LSN  (ResNet50)    | 16 | 16 x 1 x 1 |  45.7|56.1   | -
LSN+SS(ResNet50) | 16 | 16 x 2 x 1 |  50.4|62.3   | -
LSN+SS(ResNet50)    | 16 | 16 x 2 x 1 |-   |-    | 82.3
LSN+SS(ResNet50)    | 16 | 16 x 2 x 1 |-   |-    | 84.0
## Acknowledgments    
* We thank the [TSN](https://github.com/yjxiong/temporal-segment-networks) and [conv-tt-lstm](https://sites.google.com/nvidia.com/conv-tt-lstm).
## Contact Information       
Should you have any question or suggestion regarding our released code, please feel free to contact us: zhenweiwang493@gmail.com.  

