import os
import numpy as np
from torch.utils.data import Dataset
import progressbar
import matplotlib.pyplot as plt

class TrainDataSet(Dataset):
    def __init__(self, split='indiv', data_root='data/Dublin_Training_test/', DIM=9, num_point=1024, num_classe=4, test_area=5, block_size=10.0, sample_rate=0, transform=None):
        super().__init__()
        self.num_point = num_point#Définition du nombre de point dans un block
        self.block_size = block_size #!taille du bloc 
        self.transform = transform#Transformation des données 
        self.dim = DIM
        if split =='train':
            data_root = data_root + 'Training/'
        if split == "test":
            data_root = data_root + 'Test/'
        if split =='indiv':
            data_root = data_root + 'Indiv/'
        nuage =  sorted(os.listdir(data_root)) #liste [6] contenant le nom des fichier texte des nuages de points 
        nuage_split = [nuag for nuag  in nuage] #liste [6] contenant le nom des fichier texte des nuages de points 
        self.room_points, self.room_labels = [], []#Variable vide 
        self.room_coord_min, self.room_coord_max = [], []#Variable vide pour préparation 
        num_point_all = []
        labelweights = np.zeros(num_classe)#Initialise les poids par classe 
        
        for room_name in progressbar.progressbar(nuage_split):#Pour chaque ligne dans room_split (entrainement)
            room_path = os.path.join(data_root, room_name)#Récupère le chemin a parti de 'data'
            room_data = np.loadtxt(room_path)  # recupère le fichier de points au format XYZ N GRB et L
            points, labels = room_data[:, 0:self.dim], room_data[:, self.dim]  #Divise XYZRGB et L pour avoir deux matrices séparé 
            tmp, bins = np.histogram(labels, bins=num_classe)#Pour chaque classe, compte le nombre de points dans chaque classe
            labelweights += tmp#Met a jour lels poids en fcntion des classes les plus représenté dans le nuage 
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]#Coordonnées min et max
            self.room_points.append(points), self.room_labels.append(labels)#Met les nuages de chaque point a la suite 
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)#Met le min et max a la suite
            num_point_all.append(labels.size)#Nombre de point dans le nuage 
            
        labelweights = labelweights.astype(np.float32)#Convertie en float
        labelweights = labelweights / np.sum(labelweights)#Normalise les poids pour les faire comprendre entr 0 et 1
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 2.0)#Fait en sorte que le plus petit poids soit de 1
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)#Pour chaque nuage de point définie des prob selon leur nbr de pts
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)#Nombre de bloc de n_points 
        room_idxs = []
        for index in range(len(nuage_split)):#pour chaque nuage de points 
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))#Liste les itération avec leur numéro de nuage de points 
            #­653 itération pour 25 nuage au total
        self.room_idxs = np.array(room_idxs)#5Convertion en array numpy
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))#Compte le nombre d'échantillon totale pris en compte 
#Tous les nuages de points sont divisé et stocker dans les différentes liste room : label, point, idx 
    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]#Numéro du nuage a travailler 
        points = self.room_points[room_idx]   # N * 6, XYZ et RGB du nuage de points
        labels = self.room_labels[room_idx]   # N Label du nuage correspond a idx 
        N_points = points.shape[0]#Nombre de point dans le nuage de point
        dim=self.dim

        while (True):
            center = points[np.random.choice(N_points)][:3]#☺Choisi un point au hazard dans le nuage 
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]#Minimum du bloc 
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]#Maximum du bloc 
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #Recupère tous les id de point compris dans le bloc min et max qui est définie (dépend du block size)
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:#Permet de choisir 4096 points 
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)#Si le point_idx est supérieur a 4096 réchantilone 
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)#Si le point_idx et inférieur remplir aléatoirement 

        # normalize
        self.selected_points = points[selected_point_idxs, :]  # On récupère les points des indexes precedement selectionner
        self.current_points = np.zeros((self.num_point, dim))  # num_point * 6
        # self.current_points[:, dim-3] = self.selected_points[:, 0] / self.room_coord_max[room_idx][0]
        # self.current_points[:, dim-2] = self.selected_points[:, 1] / self.room_coord_max[room_idx][1]
        # self.current_points[:, dim-1] = self.selected_points[:, 2] / self.room_coord_max[room_idx][2]
        self.selected_points[:, 0] = self.selected_points[:, 0] - center[0]
        self.selected_points[:, 1] = self.selected_points[:, 1] - center[1]
        self.selected_points[:, 2] = self.selected_points[:, 2] - center[2]
        #self.selected_points[:, 6:9] /= 255.0
        self.current_points[:, 0:dim] = self.selected_points #point normalisé 
        self.current_labels = labels[selected_point_idxs] #label des points normalisé
        
       
       
        name1=(str(idx)+'Point.txt')
        name2=(str(idx)+'Label.txt')
        name_ctr=(str(idx)+' - Centre.txt')
        fmt='%1.3f'
        center_t=center.reshape(1,-1)
        path=os.getcwd()
        # os.chdir(path +'/Récupération')
        #datafinal=np.c_[self.current_points, self.current_labels]
        np.savetxt('Récupération/Training/'+name_ctr, center_t, fmt)
        np.savetxt('Récupération/Training/'+name1, self.current_points, fmt)
        np.savetxt('Récupération/Training/'+name2, self.current_labels, fmt)
        datafinal=np.c_[self.current_points, self.current_labels]
        np.savetxt('Récupération/Training/'+'Total.txt', datafinal, fmt)
        
        if self.transform is not None:#Pour transformer les point si l'on veut
            self.current_points, self.current_labels = self.transform(self.current_points, self.current_labels)
        return self.current_points, self.current_labels#retourne en sortie current label et current points

    def __len__(self):
        return len(self.room_idxs)

class DatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, DIM=9, stride=0.5, block_size=1.0, padding=0.001, num_classe=6):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.dim=DIM
        self.scene_points_num = []
        if self.split == 'indiv':
            root = root + 'Indiv/'
            self.file_list = os.listdir(root)
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.loadtxt(root + file)#Charge le nuage
            points = data[:, :3]#Recupère les coordonnée XYZ
            self.scene_points_list.append(data[:, :self.dim])#Fait une liste de tous les nuages du dossier et importe les points 
            self.semantic_labels_list.append(data[:, self.dim-1])#Fait une liste de tous les label du dossier dans uen liste 
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)#Liste des 
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_classe)
        for seg in self.semantic_labels_list:#Pour chaque nuage de point 
            tmp, _ = np.histogram(seg, range(num_classe+1))#Compte la représentation de chaque classe dans le nuage de point  
            self.scene_points_num.append(seg.shape[0])#Nombre de point dans le nuage 
            labelweights += tmp#Definie les poids
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)#Normalise les poids 
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)#Le poids le pus petit estegale a 1

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]#Recupère le nuage de points 
        points = point_set_ini[:,:self.dim]
        labels = self.semantic_labels_list[index]#Recupère les label associé
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]#les coordonnées min et max du nuage 
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)#Permet de définir 
        #le nombre de fois que nous divison le nuage de point en X selon la taille du bloc 
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        #Nombre de pas de la grille en Y pour le découpage 
        #Définie le nombre de bloc en X et Y 
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
        #Définie data room 
        #Label rool
        #Weight
        #Index room 
        for index_y in (range(0, grid_y)):
            print("Grille Y numéro : %d sur %f"% (index_y, grid_y))
            for index_x in progressbar.progressbar(range(0, grid_x)):#Pour chaque pas de la grille en X et Y grace a la double boucle 
            #pour chaque bloc 
                s_x = coord_min[0] + index_x * self.stride #Min du bloc en X
                e_x = min(s_x + self.block_size, coord_max[0])#Max du bloc en X
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride#Min du bloc 
                e_y = min(s_y + self.block_size, coord_max[1])#Max du bloc 
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                #Récupère les point corréspondant au bloc 
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))#Nombe de mini bloc de 4096 pour chaque pas de la grille
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True#Definie si la mini grille et correcte 
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)#Total pour avoir un multiple rond de la mini grille 
                #Si lagrille contient 25887 point le plus proche et 4096*7 = 28672 donc on rajoute 28672-25887 points = 2785
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))#On rajoute la liste de points
                np.random.shuffle(point_idxs)#Aléatoire 
                data_batch = points[point_idxs, :]#Récupère les coordonnées aveec idx correspondant 
                #normlized_xyz = np.zeros((point_size, 3))#Normalise initi
                # normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]#Normalise 
                # normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                # normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                #data_batch[:, 6:9] /= 255.0#Normalise les couleur 
                #data_batch = np.concatenate((data_batch), axis=1)#Regroupe point et normaliser 
                label_batch = labels[point_idxs].astype(int)#Recupère les label 
                batch_weight = self.labelweights[label_batch]#Recupère les poids 

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch#Ajoute les ligne a l'itération précédente des point  
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch#Ajoute a la suite 
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight#Ajoute a la suite 
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs#Ajoute a la suite 
                #Empile les données pour avoir (168*138*4096) ligne et ainsi avoir un bloc pour chaque ligne 
                # name=(str(index_x)+' - '+str(index_y)+'.txt')
                # path=os.getcwd()
                # # os.chdir(path +'/Récupération')
                # datafinal=np.c_[points[point_idxs, :], label_batch]
                # print(np.shape(datafinal))
                # np.savetxt('Récupération/Indiv/'+name, datafinal, fmt='%1.3f')
                
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))        
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()