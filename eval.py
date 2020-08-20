import torch
import numpy as np
import argparse
import os
import timeit
from torch.utils.tensorboard import SummaryWriter
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import os, csv
#from test_check import *


from models import DEMUCS
from utils import print_time,import_data,loss_plot,acc_plot,eval_scores


np.random.seed(12345)
try:
    torch.backends.cudnn.benchmark = True
except:
    print('cudnn.benchmark not available')

###############################################################################
### PARSE SETTINGS ; sr = 16000Hz is fixed and data is preprocessed accordingly

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id',type=int,default=0)
parser.add_argument('--mname',type=str,default='scratchJNDdefault')
parser.add_argument('--model_name',type=str,default='m15')
parser.add_argument('--epochs',type=int,default=2000)
parser.add_argument('--bs',type=int,default=4)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--wdec',type=float,default=0.) # weight decay for optimizer
parser.add_argument('--nconv',type=int,default=14) # lossnet convolution depth
parser.add_argument('--nchan',type=int,default=32) # first channel dimension, to be doubled every 5 layers
parser.add_argument('--dist_dp',type=float,default=0.25) # droupout ratio in lossnet
parser.add_argument('--dist_act',type=str,default='no') # 'no' or 'sig' or 'tanh' or 'tshrink' or 'exp'
parser.add_argument('--ndim0',type=int,default=16) # first hidden size of the classifier
parser.add_argument('--ndim1',type=int,default=6) # second hidden size of the classifier
parser.add_argument('--classif_dp',type=float,default=0.05) # droupout ratio in classifnet
parser.add_argument('--classif_BN',type=int,default=0) # 1 if classifnet with batch-norm on 1st layer / 2 if on both hidden layers
parser.add_argument('--classif_act',type=str,default='no') # 'no' or 'sig' or 'tanh'
parser.add_argument('--Lsize',type=int,default=16384) # input signal size of lossnet
parser.add_argument('--shift',type=int,default=0) # 1 if randomly shifting signals to encourage shift invariance
parser.add_argument('--randgain',type=int,default=0) # 1 if randomly applying gain on training data to encourage amplitude invariance
parser.add_argument('--sub',type=int,default=-1) # -1 or an index to select a single perturbation subset to train on
parser.add_argument('--sub2',type=int,default=-1) # -1 or an additional index to select a single perturbation subset to train on
parser.add_argument('--minit',type=int,default=0) # 0 is using default pytorch setting, 1 is using random normal init
parser.add_argument('--opt',type=str,default='adam') # 'adam' or 'rmsp'
parser.add_argument('--dummy_test',type=int,default=0) # 0 is using default pytorch setting, 1 is using random normal init
parser.add_argument('--output_folder',type=str,default='m_example') # 0 is using default pytorch setting, 1 is using random normal init
parser.add_argument('--use_npy',type=int,default=1) # 0 is using default pytorch setting, 1 is using random normal init
parser.add_argument('--avg_channel',type=str,default='yes') # 1 is using default, 0 for averaging over time only
parser.add_argument('--warping',type=int,default=0) # 0 no warping, 1 warping
parser.add_argument('--l2_normalise',type=str,default='no')
parser.add_argument('--much_warping',type=float,default=0.50)
parser.add_argument('--dilated_conv',type=str,default='no')
parser.add_argument('--load_checkpoint',type=int,default=1)
parser.add_argument('--file_load',type=str,default='summaries/m15/scratchJNDdefault_best_model.pth')
parser.add_argument('--audio_inputs_normalise',type=int,default=0)
parser.add_argument('--ksz',type=int,default=3) # filter size for the SE model
parser.add_argument('--loss',type=str,default='FL')
parser.add_argument('--load_dfl',type=str,default='../pytorch_model/summaries/m4/scratchJNDdefault_best_model.pth')

args = parser.parse_args()

GPU_id = args.GPU_id
mname = args.mname
device = torch.device("cuda:{}".format(GPU_id) if torch.cuda.is_available() else "cpu")
print(device)

epochs = args.epochs
batch_size = args.bs
lr = args.lr
wdec = args.wdec

lr_step = 50
lr_decay = 0.98

print('optimizer with batch_size,lr,wdec,lr_step,lr_decay = ',batch_size,lr,wdec,lr_step,lr_decay)

print('\nTRAINING '+mname+' for epochs,batch_size,lr')
print(epochs,batch_size,lr)

data_path = './data/'
subsets = ['dataset_combined','dataset_eq','dataset_linear','dataset_reverb']
###############################################################################
### DATA SETTINGS AND IMPORT

Lsize = args.Lsize
print('audio input size at training == ',Lsize)
# shorter segments are discarded ; longer segments are chunked in multiples of Lsize
shift = args.shift
n_shift = 4000
if shift==1:
    print('at training, xref or xper can be randomly shifted by '+str(n_shift)+' samples ~ ',n_shift/16000)

randgain = args.randgain
if randgain==1:
    gainmin = 0.1
    gainmax = 0.8 # scaling the input range ~ [-1.25,1.25] in [-1,1]
    print('at training, for every forward, apply random gain to [xref,xper] between ',gainmin,gainmax)
    print('test data is loaded with random gains, kept fixed throughout the training')
    rgains = [gainmin,gainmax]
else:
    rgains = False
#train_loader,test_loader,train_refloader,test_refloader = import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=rgains)
'''
if args.use_npy==0:
    train_loader,test_loader,train_refloader,test_refloader = import_data(Lsize,batch_size,train_ratio=0.8,dummy_test=args.dummy_test,audio_inputs_normalise=args.audio_inputs_normalise)
else:
    train_loader,test_loader,train_refloader,test_refloader = import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=rgains,dummy_test=args.dummy_test,audio_inputs_normalise=args.audio_inputs_normalise)
'''
###############################################################################
### BUILD MODEL

nconv = args.nconv
nchan = args.nchan
dist_dp = args.dist_dp
ksz = args.ksz
minit = args.minit
print('\nBUILDING with settings nconv,nchan,dist_dp,dist_act,ndim,classif_dp,classif_BN,classif_act,minit')
print(nconv,nchan,dist_dp,minit)

model = DEMUCS(dev=device)

if args.load_checkpoint==1:
    state = torch.load(args.file_load,map_location="cpu")['state']
    model.load_state_dict(state)

for name,p in model.named_parameters():
    if p.requires_grad:
         print(name,p)

model.to(device)
###############################################################################
### EVALUATION

model.eval()

def load_noisy_data_list(valfolder = ''):#check change path names

    sets = ['val']
    dataset = {'val': {}}
    datafolders = {'val': valfolder}

    
    for setname in sets:
        foldername = datafolders[setname]

        dataset[setname]['innames'] = []
        dataset[setname]['shortnames'] = []

        filelist = os.listdir("%s"%(foldername))
        filelist = [f for f in filelist if f.endswith(".wav")]
        for i in tqdm(filelist):
            dataset[setname]['innames'].append("%s/%s"%(foldername,i))
            dataset[setname]['shortnames'].append("%s"%(i))

    return dataset['val']


# DATA LOADING - LOAD FILE DATA
def load_noisy_data(valset):

    for dataset in [valset]:

        dataset['inaudio']  = [None]*len(dataset['innames'])

        for id in tqdm(range(len(dataset['innames']))):

            if dataset['inaudio'][id] is None:
                fs, inputData  = wavfile.read(dataset['innames'][id])
                
                inputData  = np.reshape(inputData, [1, -1])
                shape = inputData.shape[1]
                
                if shape%1024==0:
                    dataset['inaudio'][id]  = np.float32(inputData)
                else:
                    new_size = ((shape//1024)+1)*1024
                    a=(np.zeros(new_size-shape))
                    inputData  = np.reshape(inputData, [-1])
                    inputData = np.append(a,inputData,axis=0)
                    inputData  = np.reshape(inputData, [1,-1])
                    dataset['inaudio'][id]  = np.float32(inputData)
                #print(inputData.shape)
                if id==15 or id==16 or id==17:
                    print(inputData.shape)
    return valset

valfolder = '/n/fs/percepaudio/PerceptualMetricsOfAudio/se_code/dataset/valset_noisy'
valset = load_noisy_data_list(valfolder = valfolder)
valset = load_noisy_data(valset)

os.mkdir("denoised/%s_denoised" %args.model_name)

for id in tqdm(range(0, len(valset["innames"]))):

    i = id # NON-RANDOMIZED ITERATION INDEX
    inputData = valset["inaudio"][i] # LOAD DEGRADED INPUT
    
    audio1 = torch.from_numpy(inputData).float().to(device)
    
    with torch.no_grad():
        
        enhanced,_ = model.forward(audio1,audio1)
        output = np.reshape(enhanced.cpu().detach().numpy(), -1)
        wavfile.write("denoised/%s_denoised/%s" % (args.model_name,valset["shortnames"][i]), 16000, output)