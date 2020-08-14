import torch
import numpy as np
import argparse
import os
import timeit
from torch.utils.tensorboard import SummaryWriter
import librosa

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
parser.add_argument('--load_checkpoint',type=int,default=0)
parser.add_argument('--file_load',type=str,default='summaries/m22/scratchJNDdefault_best_model.pth')
parser.add_argument('--audio_inputs_normalise',type=int,default=0)
parser.add_argument('--ksz',type=int,default=3) # filter size for the SE model
parser.add_argument('--loss',type=str,default='L1')
parser.add_argument('--load_dfl',type=str,default='/n/fs/percepaudio/PerceptualMetricsOfAudio/contrastive_learning/accoustic_similarity/training/summaries/m20/scratchJNDdefault_best_model.pth')
parser.add_argument('--ratio',type=float,default=0.5)

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

if args.use_npy==0:
    train_loader,test_loader,train_refloader,test_refloader = import_data(Lsize,batch_size,train_ratio=0.8,dummy_test=args.dummy_test,audio_inputs_normalise=args.audio_inputs_normalise)
else:
    train_loader,test_loader,train_refloader,test_refloader = import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=rgains,dummy_test=args.dummy_test,audio_inputs_normalise=args.audio_inputs_normalise)

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

if args.loss=='FL' or args.loss=='L1_FL' or args.loss=='L2_FL':
    state = torch.load(args.load_dfl,map_location="cpu")['state']
    model_dict = model.state_dict()
    pretrained_dict = {k.replace('module.',''): v for k, v in state.items()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict,strict=False)
    
if args.load_checkpoint==1:
    state = torch.load(args.file_load,map_location="cpu")['state']
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state.items() if 'model_se' in k}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict,strict=False)
    
model.to(device)
model.train()
'''
for name, p in model.named_parameters():
    if ".encoder." in name or ".ebatch." in name:
        p.requires_grad = False
'''
for name, p in model.named_parameters():
    if "base_encoder" in name:
        p.requires_grad = False

for name,p in model.named_parameters():
    if p.requires_grad:
        print(name,p)

if args.opt=='rmsp':
    print('optimizer == RMSprop')
    optimizer = torch.optim.RMSprop(model.parameters(),lr=lr,weight_decay=wdec)
if args.opt=='adam':
    print('optimizer == ADAM')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wdec)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,gamma=lr_decay)

###############################################################################
### PRE-CHECKS

for _, minibatch in enumerate(train_loader):
    break
model.grad_check(minibatch,optimizer,loss_fn=args.loss)


model.eval()
train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device,loss_fn=args.loss,ratio=args.ratio)
epoch_log = [0]

###############################################################################
### TRAINING

model.train()
model._modules['base_encoder'].eval()

mpath = 'summaries/'+args.output_folder+'/'
os.makedirs(mpath)

loss_log = np.zeros((epochs,2)) # train/test losses
loss_log_conv = np.zeros((epochs,2))
if args.loss=='FL' or args.loss=='L1_FL' or args.loss=='L2_FL':
    loss_log_pp = np.zeros((epochs,2))

itr = 0

start_time = timeit.default_timer()

val_loss=[]

writer = SummaryWriter(str(mpath))

for epoch in range(epochs):
    
    #### training step
    model.train()
    model._modules['base_encoder'].eval()
    n_mb = 0
    
    if args.loss=='FL':
        pp_loss = 0
    elif args.loss=='L1_FL' or args.loss=='L2_FL':
        pp_loss = 0
        conv_loss = 0
    else:
        conv_loss = 0
    
    ep_loss = torch.tensor([0.]).to(device,non_blocking=True)
    
    for _, minibatch in enumerate(train_loader):
        
        xref = minibatch[0].to(device,non_blocking=True)
        xper = minibatch[1].to(device,non_blocking=True)
        
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xref = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xref),dim=1)[:,:-n_shift]
            else:
                xref = torch.cat((xref,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xper = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xper),dim=1)[:,:-n_shift]
            else:
                xper = torch.cat((xper,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        if randgain==1:
            g = torch.zeros(batch_size).to(device,non_blocking=True)
            torch.nn.init.uniform_(g,gainmin,gainmax)
            xref = xref*g.unsqueeze(1)
            xper = xper*g.unsqueeze(1)
        
        _,loss12 = model.forward(xref,xper,loss_fn=args.loss)
        
        if args.loss=='FL':
            pp_loss+= loss12.cpu().detach().numpy()
        elif args.loss=='L1_FL' or args.loss=='L2_FL':
            conv_loss+=abcd[0].cpu().detach().numpy()
            pp_loss+= abcd[1].cpu().detach().numpy()
        elif args.loss=='L1' or args.loss=='L2':
            conv_loss+=loss12.cpu().detach().numpy()
        
        optimizer.zero_grad()
        loss12.backward()
        optimizer.step()
        ep_loss+=loss12
        
        n_mb+=1
        itr+=1
    
    loss_log[epoch,0] = ep_loss.item()/n_mb
    
    if args.loss=='FL':
        loss_log_pp[epoch,0] = pp_loss/n_mb
    elif args.loss=='L1_FL' or args.loss=='L2_FL':
        loss_log_conv[epoch,0] = conv_loss/n_mb
        loss_log_pp[epoch,0] = pp_loss/n_mb
    elif args.loss=='L1' or args.loss=='L2':
        loss_log_conv[epoch,0] = conv_loss/n_mb
    
    #### testing step
    model.eval()
    n_mb = 0
    ep_loss = torch.tensor([0.]).to(device,non_blocking=True)
    if args.loss=='FL':
        pp_loss = 0
    elif args.loss=='L1_FL' or args.loss=='L2_FL':
        pp_loss = 0
        conv_loss = 0
    else:
        conv_loss = 0
    
    with torch.no_grad():
        for _,minibatch in enumerate(test_loader):
            xref = minibatch[0].to(device,non_blocking=True)
            xper = minibatch[1].to(device,non_blocking=True)
            
            _,loss12 = model.forward(xref,xper,loss_fn=args.loss)
            
            # if n_mb<=10:
            #    writer.add_audio("noisy/sample_%d.wav" % n_mb, xref.reshape(-1), 0, sample_rate=16000)
            #    writer.add_audio("clean/sample_%d.wav" % n_mb, xper.reshape(-1), 0, sample_rate=16000)
            #    writer.add_audio("enhanced/sample_%d.wav" % n_mb, enhanced.reshape(-1), 0, sample_rate=16000)
            
            ep_loss+=loss12
            if args.loss=='FL':
                pp_loss+= loss12.cpu().detach().numpy()
            elif args.loss=='L1_FL' or args.loss=='L2_FL':
                conv_loss+=abcd[0].cpu().detach().numpy()
                pp_loss+= abcd[1].cpu().detach().numpy()
            elif args.loss=='L1' or args.loss=='L2':
                conv_loss+=loss12.cpu().detach().numpy()
            n_mb+=1
    
    loss_log[epoch,1] = ep_loss.item()/n_mb
    
    writer.add_scalar("loss/train", loss_log[epoch,0], epoch)
    writer.add_scalar("loss/valid", loss_log[epoch,1], epoch)
    
    if args.loss=='FL':
        loss_log_pp[epoch,1] = pp_loss/n_mb
        writer.add_scalar("loss/train_pp", loss_log_pp[epoch,0], epoch)
        writer.add_scalar("loss/valid_pp", loss_log_pp[epoch,1], epoch)
    elif args.loss=='L1_FL' or args.loss=='L2_FL':
        loss_log_conv[epoch,1] = conv_loss/n_mb
        loss_log_pp[epoch,1] = pp_loss/n_mb
        
        writer.add_scalar("loss/train_conventional", loss_log_conv[epoch,0], epoch)
        writer.add_scalar("loss/valid_conventional", loss_log_conv[epoch,1], epoch)
        
        writer.add_scalar("loss/train_pp", loss_log_pp[epoch,0], epoch)
        writer.add_scalar("loss/valid_pp", loss_log_pp[epoch,1], epoch)
    elif args.loss=='L1' or args.loss=='L2':
        loss_log_conv[epoch,1] = conv_loss/n_mb
        writer.add_scalar("loss/train_conventional", loss_log_conv[epoch,0], epoch)
        writer.add_scalar("loss/valid_conventional", loss_log_conv[epoch,1], epoch)
    
    val_loss.append(loss_log[epoch,1])
    if epoch>1:
        if min(val_loss[:-1]) >= val_loss[-1]:
            states = {'epochs':epochs,'state':model.state_dict(),'optim':optimizer.state_dict()}
            torch.save(states,mpath+mname+'_best_model'+'.pth')
    else:
        states = {'epochs':epochs,'state':model.state_dict(),'optim':optimizer.state_dict()}
        torch.save(states,mpath+mname+'_best_model'+'.pth')
    
    if (epoch+1)%3==0:
        print('\n***  '+mname+' -  EPOCH #'+str(epoch+1)+' out of '+str(epochs)+' current itr=',itr)
        print('averaged training loss',loss_log[epoch,0])
        print('averaged test loss',loss_log[epoch,1])
        train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device,report=False,loss_fn=args.loss,ratio=args.ratio)
        
        epoch_log.append(epoch+1)
        
        plot_name = mpath+'loss_plot'
        loss_plot(plot_name,loss_log)
        if args.loss=='FL':
            plot_name = mpath+'loss_perceptual_plot'
            loss_plot(plot_name,loss_log_pp)
        #plot_name = mpath+'acc_plot'
        #acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log)
        
        print_time(timeit.default_timer()-start_time)
    
    scheduler.step()
    
    
    ## Evaluate the test dataset at given intervals
    '''
    if (epoch+1)%3==0:
        # triplet names=["mymetric","fftnet","bwe"]
        a = run_correlation_triplets(args,args.output_folder,mpath+mname+'_best_model'+'.pth',soundfile_used=args.audio_inputs_normalise)
        writer.add_scalar("test_triplet/mymetric", a[0], epoch)
        writer.add_scalar("test_triplet/fftnet", a[1], epoch)
        writer.add_scalar("test_triplet/bwe", a[2], epoch)
     
    if (epoch+1)%3==0:
        # MOS names=["voco","fftnet","bwe","jiaqi"]
        a = run_correlation_MOS(args,args.output_folder,mpath+mname+'_best_model'+'.pth',soundfile_used=args.audio_inputs_normalise)
        writer.add_scalar("test_MOS/voco", a[0], epoch)
        writer.add_scalar("test_MOS/fftnet", a[1], epoch)
        writer.add_scalar("test_MOS/bwe", a[2], epoch)
        writer.add_scalar("test_MOS/jiaqi", a[3], epoch)
    '''
###############################################################################
#### POST-TRAINING save and export

print('\nTRAINING FINISHED for model '+mname+'\n')

for g in optimizer.param_groups:
    lr_end = g['lr']
print('\nlr_end == ',lr_end)
print_time(timeit.default_timer()-start_time)
print('#iter = ',itr)

plot_name = mpath+'loss_plot'
loss_plot(plot_name,loss_log)
if args.loss=='FL':
    plot_name = mpath+'loss_perceptual_plot'
    loss_plot(plot_name,loss_log_pp)

model.eval()
print('\n\nREPORT for model '+mname)
train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device,loss_fn=args.loss,ratio=args.ratio)
#train_acc_log.append(train_acc)
#test_acc_log.append(test_acc)
epoch_log.append(epochs)
plot_name = mpath+'acc_plot'
#acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log)

states = {'epochs':epochs,'state':model.state_dict(),'optim':optimizer.state_dict(),'itr':itr,\
          'train_loss':train_loss,'test_loss':test_loss}
torch.save(states,mpath+mname+'_final_model.pth')
np.save(mpath+'losses.npy',loss_log)