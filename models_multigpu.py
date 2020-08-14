import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class base_encoder_resnet(nn.Module):
    def __init__(self,dp=0.1,dist_act='no',num_residuals=2):
        
        super(base_encoder_resnet, self).__init__()
        self.nconv = 14
        self.dist_act = dist_act
        # resnet18
        
        b1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),nn.BatchNorm1d(32), nn.ReLU(),nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        
        b2 = nn.Sequential(*resnet_block(32, 32, num_residuals, first_block=True))
        
        b3 = nn.Sequential(*resnet_block(32, 64, num_residuals))
        
        b4 = nn.Sequential(*resnet_block(64, 128, num_residuals))
        
        b5 = nn.Sequential(*resnet_block(128, 256, num_residuals))
        
        b6 = nn.Sequential(*resnet_block(256, 512, num_residuals))
        
        b7 = nn.Sequential(*resnet_block(512, 1024, num_residuals))
        '''
        b7 = resnet_block(64, 128, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(128),requires_grad=True))
        
        b8 = resnet_block(128, 128, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(128),requires_grad=True))
        
        b9 = resnet_block(128, 128, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(128),requires_grad=True))
        
        b10 = resnet_block(128, 256, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(256),requires_grad=True))
        
        b11 = resnet_block(256, 256, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(256),requires_grad=True))
        
        b12 = resnet_block(256, 256, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(256),requires_grad=True))
        
        b13 = resnet_block(256, 512, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(512),requires_grad=True))
        
        b14 = resnet_block(512, 512, num_residuals)
        self.chan_w.append(nn.Parameter(torch.randn(512),requires_grad=True))
        
        self.convs.append(nn.Sequential(*b1))
        self.convs.append(nn.Sequential(*b2))
        self.convs.append(nn.Sequential(*b3))
        self.convs.append(nn.Sequential(*b4))
        self.convs.append(nn.Sequential(*b5))
        self.convs.append(nn.Sequential(*b6))
        self.convs.append(nn.Sequential(*b7))
        self.convs.append(nn.Sequential(*b8))
        '''
        self.net = nn.Sequential(b1, b2, b3, b4, b5, b6, b7)
        '''
        self.convs.append(nn.Sequential(*b9))
        self.convs.append(nn.Sequential(*b10))
        self.convs.append(nn.Sequential(*b11))
        self.convs.append(nn.Sequential(*b12))
        self.convs.append(nn.Sequential(*b13))
        self.convs.append(nn.Sequential(*b14))
        '''
    
    def forward(self,x):
        # xref and xper are [batch,L]
        xref = x
        xper = self.net(xref)
        x = torch.sum(xper,dim=(2))/xper.shape[2] # average by channel dimension
        
        return x


class base_encoder(nn.Module):
    def __init__(self,dev=torch.device('cpu'),n_layers=20,nefilters=64):
        super(base_encoder, self).__init__()
        self.dev = dev
        nlayers = n_layers
        
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 15
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        
        nchan = nefilters
        for i in range(self.num_layers):
            if i==0:
                chin = 1
            else:
                chin = nchan
            if (i+1)%4==0:
                nchan = nchan*2
            self.encoder.append(nn.Conv1d(chin,nchan,filter_size,padding=filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(nchan))
            self.chan_w.append(nn.Parameter(torch.ones(nchan),requires_grad=True))
            
    def forward(self,x1,x2,avg_channel=0):
        
        dist = 0
        for i in range(self.num_layers):
            x1 = self.encoder[i](x1)
            x1 = self.ebatch[i](x1)
            x1 = F.leaky_relu(x1,0.1)
            if (i+1)%4==0:
                x1 = x1[:,:,::2]
            x2 = self.encoder[i](x2)
            x2 = self.ebatch[i](x2)
            x2 = F.leaky_relu(x2,0.1)
            if (i+1)%4==0:
                x2 = x2[:,:,::2]
            
            diff = (x2-x1).permute(0,2,1) # channel last
            wdiff = diff*self.chan_w[i]
            if avg_channel==1:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1,2))/diff.shape[1]/diff.shape[2] # average by time and channel dimensions
            elif avg_channel==0:
                wdiff = torch.sum(torch.abs(wdiff),dim=(1,2))/diff.shape[1] # average by time
            dist = dist+wdiff
        
        #x = torch.sum(x,dim=(2))/x.shape[2] # average by time dimension # for Lsize=40000-> [b x 1024] for 15 layers: for 16000-> [b x 1024]
        
        return dist


class projection_head(nn.Module):
    def __init__(self,ndim=[500,250],dp=0.1,BN=1,input_size=1000):
        super(projection_head, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = input_size
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1 and ilayer==0: # only 1st hidden layer
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            elif BN==2: # the two hidden layers
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            if ilayer!=1:
                MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        self.MLP = nn.Sequential(*MLP)
        
    def forward(self,dist):
        return self.MLP(dist)


class accousticNet(nn.Module):
    def __init__(self,dev=torch.device('cpu'),encoder_layers=12,encoder_filters=24,proj_ndim=[500,250],proj_dp=0.1,proj_BN=1,num_residuals=2,encoder='unet',input_size=1000):
        super(accousticNet, self).__init__()
        self.dev = dev
        if encoder=='unet':
            self.base_encoder = base_encoder(n_layers=encoder_layers,nefilters=encoder_filters)
        elif encoder=='resnet':
            self.base_encoder = base_encoder_resnet(num_residuals=num_residuals)
        self.projection_head = projection_head(ndim=proj_ndim,dp=proj_dp,BN=proj_BN,input_size=input_size)
        
    def forward(self,x1,x2,normalise = 1):
        
        # output # [N,C] # input [N,1,Lsize]
        
        x1_proj = self.projection_head.forward(self.base_encoder.forward(x1.unsqueeze(1)))
        
        x2_proj = self.projection_head.forward(self.base_encoder.forward(x2.unsqueeze(1)))
        
        if normalise==1:
            z1 = F.normalize(x1_proj, dim=1)
            z2 = F.normalize(x2_proj, dim=1)
        
        return z1,z2
        
class accousticNet_loss(nn.Module):
    
    def __init__(self,dev=torch.device('cpu'),batch_size = 16):
        super(accousticNet_loss, self).__init__()
        
        self.dev = dev
        self.nt_xent_criterion = NTXentLoss(device = self.dev, batch_size = batch_size, use_cosine_similarity = 1)
        
    def forward(self,x1,x2,normalise = 1):
        
        loss = self.nt_xent_criterion.forward(x1, x2)
        
        return loss

class NTXentLoss(torch.nn.Module):
    
    def __init__(self, device, batch_size, use_cosine_similarity,temperature=1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs.squeeze(1), zis.squeeze(1)], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)
        
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)