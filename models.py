import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models_multigpu import base_encoder

class DEMUCS(nn.Module):
    def __init__(self,dev=torch.device('cpu'),n_layers=5,nefilters=48,bidirectional=0):
        super(DEMUCS, self).__init__()
        self.dev = dev
        nlayers = n_layers
        self.MSE = nn.MSELoss(reduction='mean')
        self.num_layers = nlayers
        self.nefilters = nefilters
        self.fft_bins = [512,1024,2048]
        self.hop_sizes = [50,120,240]
        self.window_lengths = [240,600,1200]
        filter_size = 8
        merge_filter_size = 5
        self.convs = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        self.base_encoder = base_encoder(dev=dev,n_layers=16,nefilters=64)
        self.L1 = nn.L1Loss(reduction='mean')
        self.L2 = nn.MSELoss(reduction='mean')
        echannelin = [1] + [pow(2,i)*nefilters for i in range(nlayers-1)]
        echannelout = [pow(2,i)*nefilters for i in range(nlayers)]
        
        for i in range(self.num_layers):
            if i!= self.num_layers-1:
                self.encoder.append(nn.Sequential(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=(filter_size-4)//2,stride=4),nn.BatchNorm1d(echannelout[i]),nn.LeakyReLU(0.1),nn.Conv1d(echannelout[i],2*echannelout[i],1,stride=1),nn.GLU(dim=1))) 
                self.decoder.append(nn.Sequential(nn.Conv1d(echannelout[i],2*echannelout[i],1,stride=1),nn.GLU(dim=1),nn.ConvTranspose1d(echannelout[i],echannelin[i],filter_size,padding=(filter_size-4)//2,stride=4),nn.LeakyReLU(0.1)))
            else:
                self.encoder.append(nn.Sequential(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=(filter_size-4)//2,stride=4),nn.BatchNorm1d(echannelout[i]),nn.LeakyReLU(0.1),nn.Conv1d(echannelout[i],2*echannelout[i],1,stride=1),nn.GLU(dim=1)))
                self.decoder.append(nn.Sequential(nn.Conv1d(echannelout[i],2*echannelout[i],1,stride=1),nn.GLU(dim=1),nn.ConvTranspose1d(echannelout[i],echannelin[i],filter_size,padding=(filter_size-4)//2,stride=4)))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(echannelin[i]))
        
        if bidirectional==0:
            self.middle_LSTM = nn.LSTM(echannelout[-1],echannelout[-1],2,bidirectional=False,batch_first=True)
        else:
            self.middle_LSTM = nn.LSTM(echannelout[-1],echannelout[-1]//2,2,bidirectional=True,batch_first=True)
        
    def forward(self,x1,x2,loss_fn='L1'):
        
        x=x1.unsqueeze(1)
        encoder = list()
        input = x
        #print('Encoder')
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            encoder.append(x)
            #print(x.shape)
        #print('LSTM')
        
        x,_ = self.middle_LSTM(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        #print(x.shape)
        #print('Decoder')
        
        for i in range(self.num_layers):
            #print(x.shape)
            #print(encoder[self.num_layers - i - 1].shape)
            x = x + encoder[self.num_layers - i - 1]
            x = self.decoder[self.num_layers - i - 1](x)
            x = self.dbatch[self.num_layers - i - 1](x)
            #print(x.shape)
        
        se_denoised = x
        xper = x2.unsqueeze(1)
        
        if loss_fn=='L1':
            
            loss_final = self.L1(se_denoised,xper)
            
        elif loss_fn=='L2':
            loss_final = self.L2(se_denoised,xper)
            
        elif loss_fn=='FL':
            se_denoised = se_denoised*32768
            xper = xper*32768
            dist = self.base_encoder.forward(se_denoised,xper)
            loss_final = torch.mean(dist)
        
        elif loss_fn=='demucs_loss':
            loss_final = self.L1(se_denoised,xper)/se_denoised.shape[2]
            loss_final+=self.stft_function(se_denoised.squeeze(1),xper.squeeze(1))
            #print(loss_final)
        return se_denoised,loss_final
    
    def stft_function(self,enhanced,clean):
        
        loss_fn=0
        for i in range(3):
            a = torch.stft(enhanced,self.fft_bins[i],self.hop_sizes[i],self.window_lengths[i])
            b = torch.stft(clean,self.fft_bins[i],self.hop_sizes[i],self.window_lengths[i])
            eps=1e-7
            loss_fn+=torch.norm(F.relu(b)-F.relu(a), p='fro')/torch.norm(F.relu(b), p='fro')
            loss_fn+=torch.norm(torch.log(F.relu(a)+eps)-torch.log(F.relu(b)+eps),p=1)/enhanced.shape[1]
            
        return loss_fn
            
            
            
    def grad_check(self,minibatch,optimizer,loss_fn='L2'):
        xref = minibatch[0].to(self.dev)
        xper = minibatch[1].to(self.dev)
        _,loss = self.forward(xref,xper,loss_fn=loss_fn)
        
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        _,loss = self.forward(xref,xper,loss_fn=loss_fn)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)


class classifnet(nn.Module):
    def __init__(self,ndim=[16,6],dp=0.1,BN=1,classif_act='no'):
        # lossnet is pair of [batch,L] -> dist [batch]
        # classifnet goes dist [batch] -> pred [batch,2] == evaluate BCE with low-capacity
        super(classifnet, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = 1
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1 and ilayer==0: # only 1st hidden layer
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            elif BN==2: # the two hidden layers
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        MLP.append(nn.Linear(ndim[ilayer],2))
        if classif_act=='sig':
            MLP.append(nn.Sigmoid())
        if classif_act=='tanh':
            MLP.append(nn.Tanh())
        self.MLP = nn.Sequential(*MLP)
        
    def forward(self,dist):
        return self.MLP(dist.unsqueeze(1))       


class JNDnet(nn.Module):
    def __init__(self,dev=torch.device('cpu'),n_layers=12,minit=0):
        super(JNDnet, self).__init__()
        
        self.unet = Unet(n_layers=n_layers)
        self.model_classif = classifnet()
        if minit==1:
            self.model_dist.apply(weights_init) # custom weight initialization
            self.model_classif.apply(weights_init)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.dev = dev
    
    def forward(self,xref,xper,labels,avg_channel=1):
        dist = self.unet.forward(xref,xper,avg_channel=avg_channel)
        pred = self.model_classif.forward(dist)
        
        loss = self.CE(pred,torch.squeeze(labels,-1))
        class_prob = F.softmax(pred,dim=-1)
        class_pred = torch.argmax(class_prob,dim=-1)
        return loss,dist,class_pred,class_prob
    
    def grad_check(self,minibatch,optimizer,avg_channel=1):
        xref = minibatch[0].to(self.dev)
        xper = minibatch[1].to(self.dev)
        labels  = minibatch[2].to(self.dev)
        
        loss,dist,class_pred,class_prob = self.forward(xref,xper,labels,avg_channel=avg_channel)
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        loss,dist,class_pred,class_prob = self.forward(xref,xper,labels,avg_channel=avg_channel)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)