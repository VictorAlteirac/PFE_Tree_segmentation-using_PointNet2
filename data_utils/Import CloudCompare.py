import os 
import numpy as np
import progressbar

#%%
def reduction_donnees(x):
    meanx = np.mean(x[:,0])
    meany = np.mean(x[:,1])
    meanz = np.mean(x[:,2])
    x[:,0]=x[:,0]-meanx
    x[:,1]=x[:,1]-meany
    x[:,2]=x[:,2]-meanz
    return x
#%%
liste=os.listdir()


path=os.getcwd()
ROOT_DIR = path

dim=11
#%%
'''Pour tous type de données'''
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for file in progressbar.progressbar(liste):
    os.chdir(path)
    data=np.loadtxt(file, delimiter= ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,3]
    #dataNorm=data[:,dim-2:dim+1]
    dataAll=data[:,4:dim+1]
    dataAll=np.nan_to_num(dataAll, nan=1)
    datafinal=np.c_[dataxyz, dataAll,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(file, datafinal, fmt)
    
#%%
'''Pour tous type de données'''
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for file in progressbar.progressbar(liste):
    os.chdir(path)
    data=np.loadtxt(file, delimiter= ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,dim-6]
    #dataNorm=data[:,dim-2:dim+1]
    dataAll1=data[:,3:5]
    dataAll2=data[:,6:dim+1]
    dataAll1=np.nan_to_num(dataAll1, nan=1)
    datafinal=np.c_[dataxyz, dataAll1, dataAll2,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(file, datafinal, fmt)
#%%
'''Pour des données au format XYZ RGB L'''
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for file in progressbar.progressbar(liste):
    os.chdir(path)
    data=np.loadtxt(file, delimiter= ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,6]
    dataRGB=data[:,3:6]
    datafinal=np.c_[dataxyz,dataRGB,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(file, datafinal, fmt=fmt)
#%%
'''Pour tous type de données'''
fmt='%1.3f'
for file in progressbar.progressbar(liste):
    os.chdir(path)
    data=np.loadtxt(file, delimiter= ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,3]
    dataAll=data[:,4:dim+1]
    datafinal=np.c_[dataxyz,dataAll,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(file, datafinal, fmt=fmt)
#%%
 '''Pour des données sans les Normales'''
fmt='%1.3f'
for file in progressbar.progressbar(liste):
    os.chdir(path)
    data=np.loadtxt(file, delimiter= ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,dim-3]
    dataAll=data[:,3:dim-3]
    dataN=data[:,dim-2:dim+1]
    datafinal=np.c_[dataxyz, dataN,dataAll,datalab]
    #os.chdir(path +'/Training')
    np.savetxt(file, datafinal, fmt=fmt)   
#%%
'''Pour des données au format XYZ L NxNyNz'''
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for file in progressbar.progressbar(train_list):
    os.chdir(path)
    data=np.loadtxt(file, delimiter = ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,3]
    datanorm=data[:,4:7]
    datafinal=np.c_[dataxyz, datanorm, datalab]
    #os.chdir(path +'/Test')
    np.savetxt(file, datafinal, fmt=fmt) 

#%%
'''Pour des données au format XYZ L R NxNyNz'''
fmt='%1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %1.3f \t %d'
for file in progressbar.progressbar(train_list):
    os.chdir(path)
    data=np.loadtxt(file, delimiter = ' ')
    dataxyz=reduction_donnees(data[:,:3])
    datalab=data[:,3]
    datanorm=data[:,5:8]
    dataR=data[:,4]
    datafinal=np.c_[dataxyz, datanorm, dataR, datalab]
    #os.chdir(path +'/Test')
    np.savetxt(file, datafinal, fmt=fmt) 
