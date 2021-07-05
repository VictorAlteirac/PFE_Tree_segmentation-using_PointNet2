"""
Author: Benny
Date: Nov 2019
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

#%%Nuage de point individuel

#%%def nuage_indiv(model, loader, num_class, num_part, part_seg):
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
    
    print('Segmenter')
    
    num_classe=7

    dim=11
    
    A=np.loadtxt('nuage_seg_vic.txt')
    B=np.loadtxt('VR.txt')
    
    Al=A[:,11]
    
    Bl=B[:,11]
    
    C=confusion_matrix(Al,Bl)
    
    D=C/C.sum(axis=1)
    
    '''CALCUL DU mIoU POUR CHAQUE CLASSE'''
    
    tmp1, bins = np.histogram(Al, bins=num_classe)#Nombre de point pour chaque classe
    tmp2, bins = np.histogram(Bl, bins=num_classe)#Nombre de point predi pour chaque classe
    Diag=[]
    
    for i in range(len(C)):
            Diag=np.append(Diag, C[i,i])
        
    U=tmp1+tmp2-Diag
    
    IoU=Diag/U
    mIoU=sum(IoU)/num_classe
    
    print(IoU)
    print(mIoU)

#%%

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    root = 'data/Stras Total/'
    NUM_CLASSES = 7
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    DIM=args.Dimensions

    print("start loading training data ...")
    TRAIN_DATASET =  TrainDataSet(split='train', data_root=root, num_point=NUM_POINT, DIM=DIM, num_classe=NUM_CLASSES, block_size=10, sample_rate=0.8, transform=None)
    print("start loading test data ...")
    TEST_DATASET =  TrainDataSet(split='test', data_root=root, num_point=NUM_POINT,  DIM=DIM, num_classe=NUM_CLASSES, block_size=10, sample_rate=0.8, transform=None)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    
    classifier = MODEL.get_model(NUM_CLASSES, DIM).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.eval()
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        print('Optimiseur Adam')
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        print('Optimiseur SGD')
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum#Permet d'ajuster le momentum en fonction de la loss

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    acc=[]
    lossg=[]
    lossval=[]
    accval=[]
    for epoch in range(start_epoch,args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        print('Momentum actuel :%f' %momentum)
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points[:,:, :3] = provider.rotate_point_cloud_z(points[:,:, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(),target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
        acc.insert(epoch, (total_correct / float(total_seen)))
        lossg.insert(epoch, (loss_sum / num_batches))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) )
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) )
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)) )
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            lossval.insert(epoch, (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
            accval.insert(epoch, (total_correct / float(total_seen)))
            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

    plt.figure(1)
    plt.plot(acc, 'r', label='Training Accuracy')
    plt.plot(accval, 'b', label='Valid Accuracy')
    plt.xlablel=('Epoch')
    plt.ylabel=('Accuracy')
    plt.legend()
    
    plt.figure(2)
    plt.plot(lossg, 'r', label='Training Loss')
    plt.plot(lossval, 'b', label='Valid Loss')
    plt.xlablel=('Epoch')
    plt.ylabel=('Loss')
    plt.legend()
    
    np.savetxt('Accuracy Eval', lossval, fmt='%1.4f')
    np.savetxt('Accuracy', acc, fmt='%1.4f') 
    np.savetxt('Loss Function', lossg, fmt='%1.4f') 
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

