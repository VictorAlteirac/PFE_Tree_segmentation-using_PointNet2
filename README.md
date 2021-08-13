# Tree segmentation using PointNet ++

This final study project use the neural network PointNet ++ to segment automaticaly the trees on urban area. (Title : Detection and 3D reconstruction of urban trees by segmentation of point clouds: contribution of deep learning)
![image](https://user-images.githubusercontent.com/79082220/129354647-7514d7ba-6e57-449b-ab15-a246ac8a3c37.png)

## Introduction 

Many cities are seeking an eco responsible approach. For this, models are created to simulate urban temperatures.The objective of this end of study project is to use a neural network for the automatic segmentation of urban trees. This helps to help the creation of these models.
For this, the PointNet ++ network is used. The code is retrieved from user [Yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Installation  
The latest codes are tested on Window 10, CUDA 11.1, PyTorch 1.6 and Python 3.8.
To run the codes, you also need some GPU devices. In our case, we used an NVidia RTX 2070 Super with 8go VRAM. In order, to use the GPU you need to install :

   - **Nvidia Drivers**
   - **CUDA Toolkit** : (you can find the CUDA Toolkit Archive [here](https://developer.nvidia.com/cuda-toolkit-archive). Be sure to check the CUDA Toolkit version that PyTorch currently supports. You can find that information on          PyTorch's site).
   - **CUdnn Library** (navigate again in NVIDIA's website. Choose to download the version of cuDNN that corresponds to the PyTorch-supported version of the CUDA Toolkit that you        downloaded in the last step).

## Model and modifications
The model used is PointNet ++ in MSG mode. With the following levels of abstractions: 
| Level | Points | Radius (1-2) | Points per radius | MLP radius 1 | MLP Radius 2 |
|--|--|--|--|--|--| 
| 1 |  1024 | 0.3 - 2 | 32 - 64 | 16 - 16 -32 | 32 - 32 - 64|
| 2 |  512 | 1 - 3 | 32 - 64 | 64 - 64 128 | 64 - 96 - 128|
| 3 |  256 | 2 - 5 | 32 - 64 | 128 - 196 -256 | 128 - 196 - 256|
| 4 |  128 | 4 -7  | 32 - 64 | 256 - 256 - 512 | 256 - 384 -512|

## Usage 
### Data preparation
 
The form of the training, validation and test data should be as follows:
- Three folders are created in the data directory at the root
    - The training folder contains the data that will be used to train the network 
    - The test folder contains the validation data that allows the live training to be validated 
    - The Indiv folder contains the point clouds to be segmented using a trained model. 

![Dossier](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/Data.PNG)

The point cloud files in each of these folders should be with particulary form. 
- In .txt format with the following organisation

![Format](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/TXT.PNG)

The dataloader is generalisable and works with any type of data as long as the format is respected 

The additional features can be any of them. 
In this case, in order, Roughness, radius of curvature, colours (RGB) and normals. 

### Run and train
To train a PointNet ++ model to segment : 

    Train-test_sem-seg.py
    
- To start train, you hae to run the script below with changing argument according of your computeur and your data
   - Batch size, dimensions, epoch, etc...
- After processing, the journal of training is saved ```log/sem_seg/yyyy-mm-dd_hh_mm/logs```
- The model is saved in ```log/sem_seg/yyyy-mm-dd_hh_mm/checkpoints/best_model.pth``` this model can be reused for other segmentation 

To segment any point clous with any model you can lunch the script ```Indiv PointsCloud.py```. 
You just have to change the model to use and the directory with the point cloud. 

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
