'''
pytorch implementation Adrien Bitton
link: https://github.com/adrienchaton/PerceptualAudio_pytorch
paper codes Pranay Manocha
link: https://github.com/pranaymanocha/PerceptualAudio
'''

import torch
import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('Agg') # for the server
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
import librosa

###############################################################################
### misc.

def print_time(s_duration):
    m,s = divmod(s_duration,60)
    h, m = divmod(m, 60)
    print('elapsed time = '+"%d:%02d:%02d" % (h, m, s))


def length_equalise(audio,Lsize):
    
    shape1 = audio.shape[0]
    if shape1!=Lsize:
        a=(np.zeros(Lsize-shape1))
        import random
        a1=random.randint(0,1)
        if a1==0:
            audio = np.append(a,audio,axis=0)
        else:
            audio = np.append(audio,a,axis=0)
    
    return audio

###############################################################################
### data import

def import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=False,sr=16000,dummy_test=0,audio_inputs_normalise=0):
    train_clean = []
    train_noisy = []
    
    test_clean = []
    test_noisy = []
    
    if 1==1:
        # one numpy dic per pre-processed subset of audio distortion
        # each dic entry is [first signal, second signal, human label]
        fcount = 0
        print('loading training data')
        data_dic = np.load('../SE_model/data_saved_trainset_SE.npy',allow_pickle=True,encoding="latin1")
        #print(len(data_dic))
        for fid in data_dic:
            y0 = fid[0] # first signal
            y1 = fid[1] # second signal
            if audio_inputs_normalise==1:
                y0 = np.divide(y0,32768)
                y1 = np.divide(y1,32768)
            min_len = np.min([y0.shape[0],y1.shape[0]])
            N = min_len//Lsize
            if N>0:
                train_noisy.append(y0[:N*Lsize].reshape(N,Lsize))
                train_clean.append(y1[:N*Lsize].reshape(N,Lsize))
                fcount+=1
        print('paired files amount to ',fcount)
        
        fcount = 0
        print('loading testing data')
        data_dic = np.load('../SE_model/data_saved_valset_SE.npy',allow_pickle=True,encoding="latin1")
        #print(len(data_dic))
        for fid in data_dic:
            y0 = fid[0] # first signal
            y1 = fid[1] # second signal
            if audio_inputs_normalise==1:
                y0 = np.divide(y0,32768)
                y1 = np.divide(y1,32768)
            min_len = np.min([y0.shape[0],y1.shape[0]])
            N = min_len//Lsize
            if N>0:
                test_noisy.append(y0[:N*Lsize].reshape(N,Lsize))
                test_clean.append(y1[:N*Lsize].reshape(N,Lsize))
                fcount+=1
        print('paired files amount to ',fcount)
    
    train_noisy = torch.from_numpy(np.concatenate(train_noisy)).float()
    train_clean = torch.from_numpy(np.concatenate(train_clean)).float()
   
    
    test_noisy = torch.from_numpy(np.concatenate(test_noisy)).float()
    test_clean = torch.from_numpy(np.concatenate(test_clean)).float()
    
    
    train_dataset = torch.utils.data.TensorDataset(train_noisy,train_clean)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    
    # if applying random gains, train_loader is scaled at every forward
    # but train_refloader should have a fixed pre-scaling that is consistent, as for test data
    if rgains is False:
        train_refloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
        test_dataset = torch.utils.data.TensorDataset(test_noisy,test_clean)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    else:
        print('reference trainset and test data have fixed random pre-scaling')
        g = torch.zeros(train_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        train_y0_scaled = train_y0*g.unsqueeze(1)
        train_y1_scaled = train_y1*g.unsqueeze(1)
        train_dataset_scaled = torch.utils.data.TensorDataset(train_y0_scaled,train_y1_scaled,train_labels)
        train_refloader = torch.utils.data.DataLoader(train_dataset_scaled,batch_size=batch_size,shuffle=False,drop_last=False)
        
        g = torch.zeros(test_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        test_y0_scaled = test_y0*g.unsqueeze(1)
        test_y1_scaled = test_y1*g.unsqueeze(1)
        test_dataset = torch.utils.data.TensorDataset(test_y0_scaled,test_y1_scaled,test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader,train_refloader,test_refloader


def import_data_triplet(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=False,sr=16000,dummy_test=0,audio_inputs_normalise=0):
    train_y0 = []
    train_y1 = []
    train_y2 = []
    train_labels = []
    train_margin_labels = []
    test_y0 = []
    test_y1 = []
    test_y2 = []
    test_labels = []
    test_margin_labels = []
    count_dummy=0
    if 1==1:
        # one numpy dic per pre-processed subset of audio distortion
        # each dic entry is [first signal, second signal, human label]
        fcount = 0
        if dummy_test==1:
            print('loading dummy numpy array')
            data_dic = np.load('../../PerceptualAudio_pytorch/data_dummy_saved_finetune_dummy.npy',allow_pickle=True,encoding="latin1")
            for id,fid in enumerate(data_dic):
                y0 = fid[0] # first signal
                y1 = fid[1] # second signal
                y2 = fid[2] # second signal
                lab = fid[3] # human label
                if audio_inputs_normalise==1:
                    y0 = np.divide(y0,32768)
                    y1 = np.divide(y1,32768)
                    y2 = np.divide(y2,32768)
                if Lsize>40000:
                    y0 = length_equalise(y0,Lsize)
                    y1 = length_equalise(y1,Lsize)
                    y2 = length_equalise(y2,Lsize)
                min_len = np.min([y0.shape[0],y1.shape[0],y2.shape[0]])
                N = min_len//Lsize
                if N>0:
                    if np.random.rand()>train_ratio:
                        test_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        test_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        test_y2.append(y2[:N*Lsize].reshape(N,Lsize))
                        test_labels.append(np.zeros((N,1),dtype='int')+lab)
                        if int(lab)==0:
                            test_margin_labels.append(np.zeros((N,1),dtype='int')+int(-1.0))
                        elif int(lab)==1:
                            test_margin_labels.append(np.zeros((N,1),dtype='int')+int(1.0))
                        
                    else:
                        train_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        train_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        train_y2.append(y2[:N*Lsize].reshape(N,Lsize))
                        train_labels.append(np.zeros((N,1),dtype='int')+lab)
                        if int(lab)==0:
                            train_margin_labels.append(np.zeros((N,1),dtype='int')+int(-1.0))
                        elif int(lab)==1:
                            train_margin_labels.append(np.zeros((N,1),dtype='int')+int(1.0))
                    fcount+=1
            
        else:
            print('loading numpy array')
            data_dic = np.load('../../PerceptualAudio_pytorch/data_dummy_saved_finetune.npy',allow_pickle=True,encoding="latin1")
            for fid in data_dic:
                y0 = fid[0] # first signal
                y1 = fid[1] # second signal
                y2 = fid[2] # second signal
                lab = fid[3] # human label
                if audio_inputs_normalise==1:
                    y0 = np.divide(y0,32768)
                    y1 = np.divide(y1,32768)
                    y2 = np.divide(y2,32768)
                if Lsize>40000:
                    y0 = length_equalise(y0,Lsize)
                    y1 = length_equalise(y1,Lsize)
                    y2 = length_equalise(y2,Lsize)
                min_len = np.min([y0.shape[0],y1.shape[0],y2.shape[0]])
                N = min_len//Lsize
                if N>0:
                    if np.random.rand()>train_ratio:
                        test_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        test_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        test_y2.append(y2[:N*Lsize].reshape(N,Lsize))
                        test_labels.append(np.zeros((N,1),dtype='int')+lab)
                        if int(lab)==0:
                            test_margin_labels.append(np.zeros((N,1),dtype='int')+int(-1.0))
                        elif int(lab)==1:
                            test_margin_labels.append(np.zeros((N,1),dtype='int')+int(1.0))
                    else:
                        train_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        train_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        train_y2.append(y2[:N*Lsize].reshape(N,Lsize))
                        train_labels.append(np.zeros((N,1),dtype='int')+lab)
                        if int(lab)==0:
                            train_margin_labels.append(np.zeros((N,1),dtype='int')+int(-1.0))
                        elif int(lab)==1:
                            train_margin_labels.append(np.zeros((N,1),dtype='int')+int(1.0))
                    fcount+=1
        print('paired files amount to ',fcount)
    
    train_y0 = torch.from_numpy(np.concatenate(train_y0)).float()
    train_y1 = torch.from_numpy(np.concatenate(train_y1)).float()
    train_y2 = torch.from_numpy(np.concatenate(train_y2)).float()
    train_labels = torch.from_numpy(np.concatenate(train_margin_labels)).long()
    
    test_y0 = torch.from_numpy(np.concatenate(test_y0)).float()
    test_y1 = torch.from_numpy(np.concatenate(test_y1)).float()
    test_y2 = torch.from_numpy(np.concatenate(test_y2)).float()
    test_labels = torch.from_numpy(np.concatenate(test_margin_labels)).long()
    
    train_ones = float(torch.sum(train_labels).item())
    test_ones = float(torch.sum(test_labels).item())
    
    print('train/test Lsize pairs amount to ',train_y0.shape[0],test_y0.shape[0])
    print('train/test labels == one ("different") are ',int(train_ones),int(test_ones))
    print('train/test ratio of labels == one ("different") are ',train_ones/train_y0.shape[0],test_ones/test_y0.shape[0])
    
    train_dataset = torch.utils.data.TensorDataset(train_y0,train_y1,train_y2,train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    
    # if applying random gains, train_loader is scaled at every forward
    # but train_refloader should have a fixed pre-scaling that is consistent, as for test data
    if rgains is False:
        train_refloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
        test_dataset = torch.utils.data.TensorDataset(test_y0,test_y1,test_y2,test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    else:
        print('reference trainset and test data have fixed random pre-scaling')
        g = torch.zeros(train_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        train_y0_scaled = train_y0*g.unsqueeze(1)
        train_y1_scaled = train_y1*g.unsqueeze(1)
        train_y2_scaled = train_y2*g.unsqueeze(1)
        train_dataset_scaled = torch.utils.data.TensorDataset(train_y0_scaled,train_y1_scaled,train_y2_scaled,train_labels)
        train_refloader = torch.utils.data.DataLoader(train_dataset_scaled,batch_size=batch_size,shuffle=False,drop_last=False)
        
        g = torch.zeros(test_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        test_y0_scaled = test_y0*g.unsqueeze(1)
        test_y1_scaled = test_y1*g.unsqueeze(1)
        test_y2_scaled = test_y2*g.unsqueeze(1)
        test_dataset = torch.utils.data.TensorDataset(test_y0_scaled,test_y1_scaled,test_y2_scaled,test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader,train_refloader,test_refloader



def import_data_warping(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=False,sr=16000,dummy_test=0,much_warp=0.50):
    train_y0 = []
    train_y1 = []
    train_labels = []
    test_y0 = []
    test_y1 = []
    test_labels = []
    
    count_dummy=0
    if 1==1:
        
        # one numpy dic per pre-processed subset of audio distortion
        # each dic entry is [first signal, second signal, human label]
        fcount = 0
        if dummy_test==1:
            print('loading dummy numpy array')
            data_dic = np.load('../../PerceptualAudio_pytorch/data_dummy_saved.npy',allow_pickle=True,encoding="latin1")
            for id,fid in enumerate(data_dic):
                y0 = fid[0] # first signal
                y1 = fid[1] # second signal
                lab = fid[2] # human label
                if np.random.rand()>much_warp:
                    r1 = 0.85
                    r2 = 1.15
                    strech = (r2 - r1) * np.random.rand(1) + r1
                    # strech any one file :
                    import random
                    a1=random.randint(0,1)
                    if a1==0:
                        y0 = librosa.effects.time_stretch(y0,strech)
                    else:
                        y1 = librosa.effects.time_stretch(y1,strech)
                
                # decide to time strech or not: random
                if Lsize>40000:
                    y0 = length_equalise(y0,Lsize)
                    y1 = length_equalise(y1,Lsize)
                min_len = np.min([y0.shape[0],y1.shape[0]])
                N = min_len//Lsize
                if N>0:
                    if np.random.rand()>train_ratio:
                        test_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        test_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        test_labels.append(np.zeros((N,1),dtype='int')+lab)
                    else:
                        train_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        train_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        train_labels.append(np.zeros((N,1),dtype='int')+lab)
                    fcount+=1
            #print(train_labels)
        else:
            print('loading numpy array')
            data_dic = np.load('../../PerceptualAudio_pytorch/data_saved.npy',allow_pickle=True,encoding="latin1")
            for fid in data_dic:
                y0 = fid[0] # first signal
                y1 = fid[1] # second signal
                lab = fid[2] # human label
                if np.random.rand()>much_warp:
                    r1 = 0.85
                    r2 = 1.15
                    strech = (r2 - r1) * np.random.rand(1) + r1
                    # strech any one file :
                    import random
                    a1=random.randint(0,1)
                    if a1==0:
                        y0 = librosa.effects.time_stretch(y0,strech)
                    else:
                        y1 = librosa.effects.time_stretch(y1,strech)
                        
                if Lsize>40000:
                    y0 = length_equalise(y0,Lsize)
                    y1 = length_equalise(y1,Lsize)
                
                min_len = np.min([y0.shape[0],y1.shape[0]])
                N = min_len//Lsize
                if N>0:
                    if np.random.rand()>train_ratio:
                        test_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        test_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        test_labels.append(np.zeros((N,1),dtype='int')+lab)
                    else:
                        train_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                        train_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                        train_labels.append(np.zeros((N,1),dtype='int')+lab)
                    fcount+=1
        print('paired files amount to ',fcount)
    
    train_y0 = torch.from_numpy(np.concatenate(train_y0)).float()
    train_y1 = torch.from_numpy(np.concatenate(train_y1)).float()
    train_labels = torch.from_numpy(np.concatenate(train_labels)).long()
    
    test_y0 = torch.from_numpy(np.concatenate(test_y0)).float()
    test_y1 = torch.from_numpy(np.concatenate(test_y1)).float()
    test_labels = torch.from_numpy(np.concatenate(test_labels)).long()
    
    train_ones = float(torch.sum(train_labels).item())
    test_ones = float(torch.sum(test_labels).item())
    
    print('train/test Lsize pairs amount to ',train_y0.shape[0],test_y0.shape[0])
    print('train/test labels == one ("different") are ',int(train_ones),int(test_ones))
    print('train/test ratio of labels == one ("different") are ',train_ones/train_y0.shape[0],test_ones/test_y0.shape[0])
    
    train_dataset = torch.utils.data.TensorDataset(train_y0,train_y1,train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    
    # if applying random gains, train_loader is scaled at every forward
    # but train_refloader should have a fixed pre-scaling that is consistent, as for test data
    if rgains is False:
        train_refloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=True)
        test_dataset = torch.utils.data.TensorDataset(test_y0,test_y1,test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    else:
        print('reference trainset and test data have fixed random pre-scaling')
        g = torch.zeros(train_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        train_y0_scaled = train_y0*g.unsqueeze(1)
        train_y1_scaled = train_y1*g.unsqueeze(1)
        train_dataset_scaled = torch.utils.data.TensorDataset(train_y0_scaled,train_y1_scaled,train_labels)
        train_refloader = torch.utils.data.DataLoader(train_dataset_scaled,batch_size=batch_size,shuffle=False,drop_last=False)
        
        g = torch.zeros(test_y0.shape[0])
        torch.nn.init.uniform_(g,rgains[0],rgains[1])
        test_y0_scaled = test_y0*g.unsqueeze(1)
        test_y1_scaled = test_y1*g.unsqueeze(1)
        test_dataset = torch.utils.data.TensorDataset(test_y0_scaled,test_y1_scaled,test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader,train_refloader,test_refloader

###############################################################################
### evaluation functions

def loss_plot(plot_name,loss_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('loss log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(loss_log[:,0])
    plt.subplot(2,1,2)
    plt.plot(loss_log[:,1])
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('accuracy log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(epoch_log,train_acc_log)
    plt.subplot(2,1,2)
    plt.plot(epoch_log,test_acc_log)
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def eval_scores(model,train_refloader,test_refloader,device,report=True,loss_fn='L2',ratio=0.5):
    train_pred = []
    train_labels = []
    train_dist = []
    test_pred = []
    test_labels = []
    test_dist = []
    
    with torch.no_grad():
        
        train_loss = 0
        count = 0 
        for _,minibatch in enumerate(train_refloader):
            #try:
            xref = minibatch[0].to(device)
            xper = minibatch[1].to(device)
            count+=1
            _,loss_final = model.forward(xref,xper,loss_fn=loss_fn)
            train_loss += loss_final.item()
            #except:
            #    print('passing one minibatch of evaluate train_refloader')
        train_loss /= count
        # loss is averaged in the minibatch "(reduction='mean')", then divided by the number of minibatches
        
        test_loss = 0
        count = 0 
        for _,minibatch in enumerate(test_refloader):
            try:
                xref = minibatch[0].to(device)
                xper = minibatch[1].to(device)
                count+=1
                _,loss_final = model.forward(xref,xper,loss_fn=loss_fn)
                test_loss += loss_final.item()
            except:
                print('passing one minibatch of evaluate test_refloader')
        test_loss /= count
    
    if report is True:
        print('TRAINING SET')
        print('average training loss = ',train_loss)
        #print(classification_report(train_labels, train_pred, labels=[0,1], target_names=['same','different']))
    #train_acc = accuracy_score(train_labels, train_pred)
    #print('average training accuracy = ',train_acc)
    #print('average distance for train groudtruth 0,1 = ',train_dist_0,train_dist_1)
    
    if report is True:
        print('TEST SET')
        print('average test loss = ',test_loss)
        #print(classification_report(test_labels, test_pred, labels=[0,1], target_names=['same','different']))
    #test_acc = accuracy_score(test_labels, test_pred)
    #print('average test accuracy = ',test_acc)
    #print('average distance for test groudtruth 0,1 = ',test_dist_0,test_dist_1)
    
    return train_loss,test_loss


def eval_scores_triplet(model,train_refloader,test_refloader,device,report=True,avg_channel=1):
    train_pred = []
    train_labels = []
    #train_dist = []
    test_pred = []
    test_labels = []
    #test_dist = []
    
    with torch.no_grad():
        
        train_loss = 0
        for _,minibatch in enumerate(train_refloader):
            try:
                xref = minibatch[0].to(device)
                xsample1 = minibatch[1].to(device)
                xsample2 = minibatch[2].to(device)
                labels  = minibatch[3].to(device)
                loss,class_pred = model.forward(xref,xsample1,xsample2,labels,avg_channel=avg_channel)
                labels = torch.squeeze(labels,-1)
                A0 = class_pred.cpu().numpy()
                #A0[A0 == 1] = -1
                A0[A0 == 0] = -1
                train_pred.append(A0)
                train_labels.append(labels.cpu().numpy())
                train_loss += loss.item()
                
            except:
                print('passing one minibatch of evaluate train_refloader')
        train_loss /= len(train_pred)
        # loss is averaged in the minibatch "(reduction='mean')", then divided by the number of minibatches
        
        test_loss = 0
        for _,minibatch in enumerate(test_refloader):
            try:
                xref = minibatch[0].to(device)
                xsample1 = minibatch[1].to(device)
                xsample2 = minibatch[2].to(device)
                labels  = minibatch[3].to(device)
                loss,class_pred = model.forward(xref,xsample1,xsample2,labels,avg_channel=avg_channel)
                labels = torch.squeeze(labels,-1)
                A0 = class_pred.cpu().numpy()
                #A0[A0 == 1] = -1
                A0[A0 == 0] = -1
                test_pred.append(A0)
                test_labels.append(labels.cpu().numpy())
                test_loss += loss.item()
                #test_dist.append(dist.unsqueeze(-1).cpu().numpy())
            except:
                print('passing one minibatch of evaluate test_refloader')
        test_loss /= len(test_pred)
    
    train_pred = np.concatenate(train_pred)
    train_labels = np.concatenate(train_labels)
    test_pred = np.concatenate(test_pred)
    test_labels = np.concatenate(test_labels)
    #train_dist = np.concatenate(train_dist)
    #test_dist = np.concatenate(test_dist)
    
    #train_dist_0 = np.mean(train_dist[np.where(train_labels==0)])
    #train_dist_1 = np.mean(train_dist[np.where(train_labels==1)])
    #test_dist_0 = np.mean(test_dist[np.where(test_labels==0)])
    #test_dist_1 = np.mean(test_dist[np.where(test_labels==1)])
    
    if report is True:
        print('TRAINING SET')
        print('average training loss = ',train_loss)
        print(classification_report(train_labels, train_pred, labels=[-1,1], target_names=['s1','s2']))
    train_acc = accuracy_score(train_labels, train_pred)
    print('average training accuracy = ',train_acc)
    #print('average distance for train groudtruth 0,1 = ',train_dist_0,train_dist_1)
    
    if report is True:
        print('TEST SET')
        print('average test loss = ',test_loss)
        print(classification_report(test_labels, test_pred, labels=[-1,1], target_names=['s1','s2']))
    test_acc = accuracy_score(test_labels, test_pred)
    print('average test accuracy = ',test_acc)
    #print('average distance for test groudtruth 0,1 = ',test_dist_0,test_dist_1)
    
    return train_acc,test_acc,train_loss,test_loss