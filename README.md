# Tree segmentation using PointNet ++

This final study project use the neural network PointNet ++ to segment automaticaly the trees on urban area.

## Introduction 
The objective of this end of study project is to use a neural network for the automatic segmentation of urban trees.
For this, the PointNet ++ network is used. The code is retrieved from user Yanx27.

## Installation  
The latest codes are tested on Window 10, CUDA 11.1, PyTorch 1.6 and Python 3.8.
To run the codes, you also need some GPU devices. In our case, we used an NVidia RTX 2070 Super with 8go VRAM. In order, to use the GPU you need to install
    - Nvidia Drivers
    - CUDA Toolkit : (you can find the CUDA Toolkit Archive here. Be sure to check the CUDA Toolkit version that PyTorch currently supports. You can find that information on PyTorch's site).
    - CUdnn Library (navigate again in NVIDIA's website. Choose to download the version of cuDNN that corresponds to the PyTorch-supported version of the CUDA Toolkit that you downloaded in the last step).

## Model and modifications
The model used is PointNet ++ in MSG mode. With the following levels of abstractions: 
| Level | Points | Radius (1-2) | Points per radius | MLP radius 1 | MLP Radius 2 |
|--|--|--|--|--|--| 
| 1 |  1024 | 0.3 - 2 | 32 - 64 | 16 - 16 -32 | 32 - 32 - 64|
| 2 |  512 | 1 - 3 | 32 - 64 | 64 - 64 128 | 64 - 96 - 128|
| 3 |  256 | 2 - 5 | 32 - 64 | 128 - 196 -256 | 128 - 196 - 256|
| 4 |  128 | 4 -7  | 32 - 64 | 256 - 256 - 512 | 256 - 384 -512|

## Usage 
### Segmentation 
To train a PointNet ++ model to segment : 

    Train-test_sem-seg.py
  
The form of the training, validation and test data should be as follows:
- Three folders are created in the data directory at the root
    - The training folder contains the data that will be used to train the network 
    - The test folder contains the validation data that allows the live training to be validated 
    - The Indiv folder contains the point clouds to be segmented using a trained model. 

![Dossier](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/Data.PNG)

The point cloud files in each of these folders should be with particulary form. 
- In .txt format with the following organisation

![Format](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/TXT.PNG)

The additional features can be any of them. 
In this case, in order, Roughness, radius of curvature, colours (RGB) and normals. 

## Performances
Training Performances: 
| Training Acc | Training Loss | Validation Acc. | Validation Loss |
|---------|---------|---------|---------| 
| 98.1 %| 0.068 | 95.1 % | 0.202 |

Test performances 
| IoU Leafy tree | IoU Ground  | IoU Building | IoU Other | IoU Light | IoU Car |IoU Trimmed tree | mIoU |
|---------|---------|---------|---------|---------|---------|---------|---------| 
| 97.0 %| 94.8 % | 98.4% | 62.0% | 87.1 % | 62.8 % | 97.3 % | 85.1 % |

### Visuals results
![Format](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/Image2.png)
