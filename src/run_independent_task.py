import sys,os,argparse,time
import numpy as np
import torch

import utils
from torchsummary import summary
from pathlib import Path
import os
import pandas as pd

tstart=time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--seed',type=int,default=0,help='(default=%(default)d)')
parser.add_argument('--experiment',default='',type=str,required=True,choices=['mnist2','pmnist','cifar','mixture'],help='(default=%(default)s)')
parser.add_argument('--approach',default='',type=str,required=True,choices=['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet',
                                                                            'imm-mode','sgd-restart',
                                                                            'joint','hat','hat-test','hat_mask'],help='(default=%(default)s)')
parser.add_argument('--output',default='',type=str,required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs',default=200,type=int,required=False,help='(default=%(default)d)')
parser.add_argument('--lr',default=0.05,type=float,required=False,help='(default=%(default)f)')
parser.add_argument('--parameter',type=str,default='',help='(default=%(default)s)')
parser.add_argument('--optimizer',default='sgd',type=str,required=False,choices=['sgd','adam'],help='(default=%(default)s)')
parser.add_argument('--weight_initializer',default='',type=str,required=False,choices=['xavier'],help='(default=%(default)s)')

args=parser.parse_args()
# if args.output=='':
    # args.output='./res/'+args.weight_initializer+'_'+args.experiment+'_'+args.approach+'_'+str(args.optimizer)+'_'+str(args.seed)+'_'+str(args.lr)+'_'+str(args.nepochs)+'.txt'
output_file_prefix = f"app_{args.approach}_exp_{args.experiment}_opt_{args.optimizer}_wt_init_{args.weight_initializer}_lr_{args.lr}_nepoch_{args.nepochs}"
if args.output=='':
    args.output = f"./res/{output_file_prefix}.txt"
else:
    # if not os.path.exists(args.)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.output = f"{args.output}/{output_file_prefix}.txt"


print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment=='cifar':
    from dataloaders import cifar as dataloader
elif args.experiment=='mixture':
    from dataloaders import mixture as dataloader

# Args -- Approach
if args.approach=='random':
    from approaches import random as approach
elif args.approach=='sgd':
    from approaches import sgd as approach
elif args.approach=='sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach=='sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach=='lwf':
    from approaches import lwf as approach
elif args.approach=='lfl':
    from approaches import lfl as approach
elif args.approach=='ewc':
    from approaches import ewc as approach
elif args.approach=='imm-mean':
    from approaches import imm_mean as approach
elif args.approach=='imm-mode':
    from approaches import imm_mode as approach
elif args.approach=='progressive':
    from approaches import progressive as approach
elif args.approach=='pathnet':
    from approaches import pathnet as approach
elif args.approach=='hat-test':
    from approaches import hat_test as approach
elif args.approach=='hat':
    from approaches import hat_indep as approach
elif args.approach == 'hat_mask':
    from approaches import hat_mask_indep as approach
elif args.approach=='joint':
    from approaches import joint as approach

# Args -- Network
if args.experiment=='mnist2' or args.experiment=='pmnist':
    if args.approach=='hat' or args.approach=='hat-test':
        from networks import mlp_hat as network
    else:
        from networks import mlp as network
else:
    if args.approach=='lfl':
        from networks import alexnet_lfl as network
    elif args.approach=='hat':
        from networks import alexnet_hat as network
    elif args.approach=='progressive':
        from networks import alexnet_progressive as network
    elif args.approach=='pathnet':
        from networks import alexnet_pathnet as network
    elif args.approach=='hat-test':
        from networks import alexnet_hat_test as network
    elif args.approach=='hat_mask':
        from networks import mask as network

    else:
        from networks import alexnet as network

########################################################################################################################
# Load
print('Load data...')
data,taskcla,inputsize=dataloader.get(seed=args.seed)
print('Input size =',inputsize,'\nTask info =',taskcla)

# # Inits
# print('Inits...')
# if args.approach == "hat_mask":
#     net=network.Net(inputsize,taskcla,args=args).cuda()
# else:
#     net=network.Net(inputsize,taskcla).cuda()


# utils.print_model_report(net)

# appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
# print(appr.criterion)
# utils.print_optimizer_config(appr.optimizer)
# print('-'*100)

# Loop tasks
acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
for t,ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    print('*'*100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t==0:
            xtrain=data[t]['train']['x']
            ytrain=data[t]['train']['y']
            xvalid=data[t]['valid']['x']
            yvalid=data[t]['valid']['y']
            task_t=t*torch.ones(xtrain.size(0)).int()
            task_v=t*torch.ones(xvalid.size(0)).int()
            task=[task_t,task_v]
        else:
            xtrain=torch.cat((xtrain,data[t]['train']['x']))
            ytrain=torch.cat((ytrain,data[t]['train']['y']))
            xvalid=torch.cat((xvalid,data[t]['valid']['x']))
            yvalid=torch.cat((yvalid,data[t]['valid']['y']))
            task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
            task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
            task=[task_t,task_v]
    else:
        # Get data
        xtrain=data[t]['train']['x'].cuda()
        ytrain=data[t]['train']['y'].cuda()
        xvalid=data[t]['valid']['x'].cuda()
        yvalid=data[t]['valid']['y'].cuda()
        task=t

    # Inits
    print('Inits...')
    if args.approach == "hat_mask":
        net=network.Net(inputsize,taskcla,args=args).cuda()
    else:
        net=network.Net(inputsize,taskcla).cuda()


    utils.print_model_report(net)

    appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args)
    print(appr.criterion)
    utils.print_optimizer_config(appr.optimizer)
    print('-'*100)

    # Train
    appr.train(task,xtrain,ytrain,xvalid,yvalid)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest=data[u]['test']['x'].cuda()
        ytest=data[u]['test']['y'].cuda()
        test_loss,test_acc, test_avg_reg=appr.eval(u,xtest,ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
        acc[t,u]=test_acc
        lss[t,u]=test_loss

        test_loss_file_name = "test_loss_"+Path(args.output).stem
        test_loss_file_path = f"{Path(args.output).parent}/{test_loss_file_name}.csv"
        # loss_file_path = f"./res/wt_init_{self.args.weight_initializer}_losses_app_{self.args.approach}_exp_{self.args.experiment}_opt_{self.args.optimizer}_lr_{self.lr}_nepoch_{self.nepochs}.csv"
        if os.path.exists(test_loss_file_path):
            test_loss_df = pd.read_csv(test_loss_file_path)
        else:
            test_loss_df = pd.DataFrame(columns=["Task","Train_loss","Train_acc","Avg_train_reg"])

        test_loss_df = pd.concat([test_loss_df,pd.DataFrame({"Task":f"{u} {data[u]['name']}",
                            "Train_loss":round(test_loss,6),
                            "Train_acc":round(100*test_acc,6),
                            "Avg_train_reg":round(test_avg_reg,6),
                            # "Val_loss":round(valid_loss,6),
                            # "Val_acc":round(100*valid_acc,6),
                            # "Avg_val_reg":round(valid_avg_reg,6),
                            },index=[0])])
    test_loss_df.to_csv(test_loss_file_path,index=False)
    
    # Save
    print('Save at '+args.output)
    np.savetxt(args.output,acc,'%.4f')

# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.1f}% '.format(100*acc[i,j]),end='')
    print()
print('*'*100)
print('Done!')

print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        #save task names
        from copy import deepcopy
        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t,ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
            appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
        #pickle
        import gzip
        import pickle
        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

########################################################################################################################
