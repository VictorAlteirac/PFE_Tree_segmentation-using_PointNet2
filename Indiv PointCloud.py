# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:43:59 2021

@author: altei
"""

import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import progressbar
from sklearn.metrics import confusion_matrix
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
CUDA_LAUNCH_BLOCKING=1

'''La segmentation sementique permet de segmenter des nuages de point directement dans une scene, contrairement a la part 
segmentation, la principale différence réside dans la facon de traiter le nuage de point, la zone et diviser en batch size qui 
sont diviser en carré pour le traitement du nuage'''
#classes = ['Arbre', 'Autre']#Base de données de Elena
#classes = ['Bati', 'Sol', 'Végé', 'Autre']#Pour la base de données de Dublin City 
# classes = ['Autres', 'Bati', 'Végé', 'Sol','Bagnole','Candélabre']#Pour la base de données de Maxime 
#classes = ['Poutre','Tableaux', 'Plafond', 'Chaise', 'Lumières', 'Poteaux', 'Portes','Sol', 'Tables','Murs','Fenetre']
#classes = ['Sol', 'Bati', 'Autre', 'Platane Taillé','Platane Feuillu']#Boulevard victoire phéologie
classes = ['Arbres Feuillu', 'Sol', 'Batiment', 'Autre','Candélabre','Voiture', 'Arbre Taillé']#Pour la base de données de Boulevard Victoire / Jean Jaurès
#classes = ['Autre', 'Sol', 'Marquage', 'Végé','Bati','Ligne', 'Poteau', 'Voiture', 'Cloture']
#classes = ['Tilleul','Platane taillé', 'Platane Feuillu', 'Sol', 'Batiment', 'Autre','Candélabre','Voiture']#Pour la base de données de Boulevard Victoire / Jean Jaurès
# classes = ['Autre', 'Ground', 'BAti', 'Light','Bollard','Poubelle','Barrière','Pede','Voiture','Végétation']#Pour la base de données Paris - Lille 
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat
'''Chaque classe est assigné a un numéro dans ce tableau'''


#Argument et paramètre pour l'apprentissage du réseau 
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_msg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=18, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int,  default=1, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--données', type=int, default=5, help='XYZ N --> 1 ; XYZ RGB --> 2 ; XYZ N RGB -->3 ; XYZ R N --> 4 ; XYZ --> 5')
    parser.add_argument('--Dimensions', type=int, default=11, help='Nombre de caract. du nuage = Entrée du réseau')
    return parser.parse_args()

args = parse_args()
if args.données == 1:
    from data_utils.LoaderPointNet1 import TrainDataSet, DatasetWholeScene
elif args.données == 2:
    from data_utils.LoaderPointNet2 import TrainDataSet, DatasetWholeScene
elif args.données == 3:
    from data_utils.LoaderPointNet3 import TrainDataSet, DatasetWholeScene
elif args.données == 4:
    from data_utils.LoaderPointNet4 import TrainDataSet, DatasetWholeScene
elif args.données == 5:
    from data_utils.LoaderPointNet5 import TrainDataSet, DatasetWholeScene

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b,n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


    args=parse_args()
    
    NUM_CLASSES = 7
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    DIM=args.Dimensions
    '''Important, pour trouver les segment, le classifieur a besoin de connaitre la classe du nuage de point dans la 
    label la classe doit etre rentrée dans torch.tensor([0])'''
                      
    root='data/Boulvard INSA/Boulevard Pheno/'
    

    experiment_dir = Path('./log/Repet/2021-07-02_12-32')
    str(experiment_dir)
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES, DIM).cuda()
    checkpoint=torch.load(str(experiment_dir)+'/checkpoints/best_model.pth')
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    
    TEST_DATASET_WHOLE_SCENE =  DatasetWholeScene(split='indiv', root=root, DIM=DIM, block_points=NUM_POINT, block_size=10, stride=5, num_classe=NUM_CLASSES)
    
    scene_id = TEST_DATASET_WHOLE_SCENE.file_list#Nom du fichier a traiter
    scene_id = [x[:-4] for x in scene_id]
    num_batches = len(TEST_DATASET_WHOLE_SCENE)#Nombre de fichier 
    
    for batch_idx in range(num_batches):
        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        
        for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, DIM))
                
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
                
                for sbatch in progressbar.progressbar(range(s_batch_num)):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:DIM] /= 1.0
                    
                    torch_data = torch.Tensor(batch_data)
                    torch_data= torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
                    
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],batch_pred_label[0:real_batch_size, ...],batch_smpw[0:real_batch_size, ...])

                    pred_label = np.argmax(vote_label_pool, 1)
    points=whole_scene_data
    sortie=np.insert(points,points.shape[1],pred_label, axis=1)
    sortie = np.savetxt("nuage_seg_vic.txt", sortie, fmt='%1.3f')
    # data=np.loadtxt("nuage_seg_vic.txt")