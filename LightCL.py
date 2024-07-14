import torch
from torch import optim
from torch import nn  
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import random
from prune import *
from utils import *
import torch.nn.init as init
import argparse
from models import * 
from datasets.seq_cifar10 import *
from datasets.seq_tinyimagenet import *

#-----------------------Getting Information----------------------------------
print("----------------Get Args----------------")
parser = argparse.ArgumentParser(description='Light Continual Learning')
parser.add_argument('--lr',       type=float, default=0.01,      help='learning rate (default: 0.01)')
parser.add_argument('--Beta',     type=float, default=0.0002,    help='Beta (default: 0.0002)')
parser.add_argument('--BufferNum',type=int,   default=15,        help='Buffer Num (default: 15)')
parser.add_argument('--Ratio', type=float, default=0.15,      help='Ratio (default: 0.15)')
parser.add_argument('--Seed',     type=int,   default=0,         help='Seed (default: 0)')
parser.add_argument('--pretrain', action='store_false', default=True,     help='pretrain(default: True)')
parser.add_argument('--Dataset',  type=str,   default='CIFAR10', help='Dataset (default: CIFAR10; Other: TinyImageNet)')
parser.add_argument('--Sparse', action='store_true', default=False,     help='Sparse(default: False)')

args = parser.parse_args()
Learning_rate, Beta, Buffer_num, Ratio, seed, Dataset_name = args.lr, args.Beta, args.BufferNum, args.Ratio, args.Seed, args.Dataset
NOW_NAME = f'Lr {Learning_rate}-Beta {Beta}-BufferNum {Buffer_num}-Ratio {Ratio}-Seed {seed}-Dataset {Dataset_name}-pretrain {args.pretrain}-Sparse {args.Sparse}'
print("Begin with Setting of ", NOW_NAME)

TIL_interval = 0
if Dataset_name=='CIFAR10':
    Dataset_name = 'seq-cifar10'
    TIL_interval = 2
elif Dataset_name=='TinyImageNet':
    Dataset_name = 'seq-tinyimg'
    TIL_interval = 20

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # cuDNN's auto-tuner   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Now Device is: ", device)
pretrain_model = 'ResNet18_for_LightCL.pth'

num_task = 5
epochs = 50
if Dataset_name=='seq-cifar10':
    Data_mnist = SequentialCIFAR10(None)
elif Dataset_name=='seq-tinyimg':
    Data_mnist = SequentialTinyImagenet(None)
    num_task = 10
    epochs = 100

train_loader = {}
test_loader = {}
for i in range(num_task):
    train_, test_ = Data_mnist.get_data_loaders()
    train_loader[i] = train_
    test_loader[i] = test_

print("got dataset")
#-----------------------Training Function----------------------------------
def train(model, task, optimizer, data_loader, Pruner=None):
    model.train()
    epoch_loss = 0
    if Pruner!=None: Pruner.apply(model)
    if args.pretrain==True:             #retrain part of sparse Network. Only in the first time
        determine_grad = False
        for name, param in model.named_parameters():
            if args.Sparse==True and '3.1' in name: determine_grad = True  # if sparse, retrain layer after 3.1block(included) only in task 0
            if '4_1' in name: determine_grad = True         
            param.requires_grad = determine_grad
    
    for input, target in data_loader:
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        optimizer.zero_grad()
        output, _ = model(input)
        loss = F.cross_entropy(output[:,task*TIL_interval:(task+1)*TIL_interval], target-task*TIL_interval)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        if Pruner!=None: Pruner.apply(model)
    return epoch_loss / len(data_loader)

def test(task, model, data_loader, Pruner=None):
    model.eval()
    correct_TIL, correct_CIL = 0, 0
    if Pruner!=None: Pruner.apply(model)
    for input, target in data_loader:
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        output,_ = model(input)
        correct_TIL += (F.softmax(output[:, task*TIL_interval:(task+1)*TIL_interval], dim=1).max(dim=1)[1] == (target-task*TIL_interval)).data.sum()
        correct_CIL += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()   
    return correct_TIL / len(data_loader.dataset), correct_CIL/len(data_loader.dataset)

def get_forgetting_info(model, now_task, Pruner):
    accuracy_TIL, accuracy_CIL = 0.0, 0.
    for i in range(now_task+1):
        TIL, CIL = test(i, model, test_loader[i], Pruner)
        print(f'{i}_TIL:', TIL, f'{i}_CIL:', CIL)
        accuracy_TIL+=TIL
        accuracy_CIL+=CIL
    return accuracy_TIL/(now_task+1.), accuracy_CIL/(now_task+1.)

def my_train(task, model, optimizer, data_loader, dict_features_standard=None, Buffer = None, Buffer_mask = None, Pruner = None):
    model.train()
    determine_grad = False
    for name, param in model.named_parameters():
        if '4_1' in name: determine_grad = True   
        param.requires_grad = determine_grad
        
    for input, target in data_loader:
        if Pruner != None:
            Pruner.apply(model)
        input = input.cuda(non_blocking=True)
        target= target.cuda(non_blocking=True)
        optimizer.zero_grad()
        L_activation = 0
        _, out_feature = model(Buffer[0].clone())
        for name in out_feature:
            if 'linear' in name or Ratio == 1.0:
                L_activation+= torch.sum(Beta * ((dict_features_standard[name] - out_feature[name])**2))
            else:
                tmp_mask = Buffer_mask[name].detach()
                tmp_mask = tmp_mask.unsqueeze(1).unsqueeze(1)
                tmp_mask = tmp_mask.expand(tmp_mask.size()[0], out_feature[name].size()[2], out_feature[name].size()[3])
                tmp_mask = tmp_mask.expand_as(out_feature[name])
                tmp_mask = tmp_mask.to(device)
                L_activation+= torch.sum(Beta * (((out_feature[name]-dict_features_standard[name]).mul_(tmp_mask))**2))
        
        output2, _ = model(input)
        loss = F.cross_entropy(output2[:,task*TIL_interval:(task+1)*TIL_interval], target-task*TIL_interval) + L_activation
            
        if torch.isnan(L_activation).any():
            raise ValueError("L_activation is NAN")

        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

    if Pruner!=None:
        Pruner.apply(model)

def my_process(model, Pruner=None): 
    if Pruner!=None: Pruner.apply(model)
    optimizer = optim.SGD(params=model.parameters(), lr=Learning_rate)
    Buffer = None
    Buffer_mask = {}            # mask unimportant activation/feature maps. [channels]=0/1 represent [i]channel whether is important
    sparsity_param_num = {}     # recode important channel number in each layer
    for task in range(num_task):      
        print(f'----------------{task} is beginning------------------') 
        if Pruner!=None: Pruner.apply(model)
        if task==0:
            for _ in tqdm(range(epochs)):
                train(model, task, optimizer, train_loader[task], Pruner)
        else:
            model.train()
            input, target = next(iter(train_loader[task]))
            input = input.to(device)
            target = target.to(device)
            Buffer = (input[:Buffer_num].detach(), target[:Buffer_num].detach())
            _, out_feature = model(Buffer[0])
            for name in out_feature:out_feature[name] = out_feature[name].detach()
            for now_epochs in tqdm(range(epochs)):
                my_train(task, model, optimizer, train_loader[task], out_feature, Buffer, Buffer_mask, Pruner)
                if now_epochs%5==0: get_forgetting_info(model, task, Pruner)
        
        input, _ = next(iter(train_loader[task]))
        input = input.to(device)
        model.train()
        if Pruner!=None: Pruner.apply(model)
        _, out_feature= model(input)
        for name in out_feature:
            if 'linear' in name:continue
            out_feature[name] = out_feature[name].detach()
            (sample, channels, hx, hy) = out_feature[name].size()
            now = out_feature[name].detach()
            now = now.permute(1, 0, 2, 3) #->[channel][sample_num][hx][hy]
            now = torch.abs(now.detach())
            now = now.sum(-1).sum(-1).sum(-1) #->[channel]
            n_keep = int(channels * Ratio + 0.5)
            threshold  =torch.kthvalue(now.flatten(), max(1, channels - n_keep))
            tmp = (now >= threshold[0])
            if task==0: Buffer_mask[name] = tmp
            else: Buffer_mask[name] = torch.logical_or(Buffer_mask[name], tmp)
            sparsity_param_num[name] = Buffer_mask[name].count_nonzero()/channels
            print(name, "important channel ratio is ", sparsity_param_num[name])
        print("----------Metric-----------")
        now_forget_TIL, now_forget_CIL = get_forgetting_info(model, task, Pruner)
        print(f"All forget{task}_TIL:", now_forget_TIL, f'CIL:', now_forget_CIL)

def get_pruner(model):  
    print("----------------Begin Prunning----------------")
    dict_sparsity = {}
    for name, param in model.named_parameters():
        if param.dim()<=1:continue
        if '3.0' in name or '3.1' in name:
            dict_sparsity[name] = 0.7
        elif '4_1' in name or '4_2' in name:
            dict_sparsity[name] = 0.8
        else:
            dict_sparsity[name] =0.2 #0.2 #0.1
        if args.Sparse==False:
            dict_sparsity[name]=0.

    Pruner = ChannelPruner(model, dict_sparsity)
    Pruner.apply(model)
    print("MODEL Sparsity:",get_model_sparsity(model))
    print(get_model_size(model, data_width=32, count_nonzero_only=True).item()/8/1024/1024,"MB  /", get_model_size(model, data_width=32, count_nonzero_only=False)/8/1024/1024,"MB")
    return Pruner

def get_model():
    model = None
    if args.pretrain==True:
        model = resnet18('ImageNet').to(device)
        model.load_state_dict(torch.load(pretrain_model))
        output_head = 0
        if Dataset_name == 'seq-cifar10':   output_head = 10
        elif Dataset_name == 'seq-tinyimg': output_head = 200
        new_linear = nn.Linear(512, output_head)    
        init.xavier_uniform_(new_linear.weight)
        init.zeros_(new_linear.bias)
        model.linear.weight.data = new_linear.weight.data
        model.linear.bias.data = new_linear.bias.data
        model = model.to(device)
    else:
        model = resnet18(Dataset_name).to(device)
    return model

model = get_model()
Pruner = get_pruner(model)
my_process(model, Pruner)
print("----------Final Metric-----------")
print(NOW_NAME)
now_forget_TIL, now_forget_CIL = get_forgetting_info(model, num_task-1, Pruner)
print(f"TIL:", now_forget_TIL, f'CIL:', now_forget_CIL)
