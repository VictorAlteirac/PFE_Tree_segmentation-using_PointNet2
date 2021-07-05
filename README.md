# Tree segmentation using PointNet ++

This final study project use the neural network PointNet ++ to segment automaticaly the trees on urban area.

## Introduction 
The objective of this end of study project is to use a neural network for the automatic segmentation of urban trees.
For this, the PointNet ++ network is used. The code is retrieved from user Yanx27.

## Installation  
To use the implemantation, you must have version 2.7 of Python or higher and pytorch version 1.7 on Windows 10. 
In addition, the CUDA 10.2 Toolkit is used with the Cudnn version adapted to this version of CUDA.

## Usage 
### Segmentation 
To train a PointNet ++ model to segment : 

    Train-test_sem-seg.py
  
The form of the training, validation and test data should be as follows:
- Three folders are created in the data directory at the root

![prediction example](https://github.com/VictorAlteirac/PFE_Tree_segmentation-using_PointNet2/blob/main/Image/Data.PNG)
