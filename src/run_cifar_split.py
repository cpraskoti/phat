import sys,os,argparse,time
import numpy as np
import torch

import utils
from torchsummary import summary
from pathlib import Path
import pandas as pd
import os
import itertools
import random
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
parser.add_argument('--datatype',default='cifar',type=str,required=False,choices=['cifar','cifar10','cifar100','mnist2',"mnist5"],help='(default=%(default)s)')

args=parser.parse_args()
# if args.output=='':
    # args.output='./res/'+args.weight_initializer+'_'+args.experiment+'_'+args.approach+'_'+str(args.optimizer)+'_'+str(args.seed)+'_'+str(args.lr)+'_'+str(args.nepochs)+'.txt'

# output_file_prefix = f"app_{args.approach}_exp_{args.experiment}_opt_{args.optimizer}_wt_init_{args.weight_initializer}_lr_{args.lr}_nepoch_{args.nepochs}"
# if args.output=='':
#     args.output = f"./res/{output_file_prefix}.txt"
# else:
#     # if not os.path.exists(args.)
#     Path(args.output).mkdir(parents=True, exist_ok=True)
#     args.output = f"{args.output}/{output_file_prefix}.txt"


print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':',getattr(args,arg))
print('='*100)

########################################################################################################################

# Seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else: print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment

if args.experiment=='mnist2' and args.datatype == "mnist5":
    from dataloaders import mnist5 as dataloader

elif args.experiment=='mnist2':
    from dataloaders import mnist2 as dataloader

elif args.experiment=='pmnist':
    from dataloaders import pmnist as dataloader

elif (args.experiment=='cifar') and (args.datatype=='cifar10'):
    from dataloaders import cifar_10 as dataloader
elif (args.experiment=='cifar') and (args.datatype=='cifar100'):

    from dataloaders import cifar_100 as dataloader
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
    from approaches import hat as approach
elif args.approach == 'hat_mask':
    from approaches import hat_mask as approach
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

# ########################################################################################################################
# # Load
# print('Load data...')
# data,taskcla,inputsize=dataloader.get(seed=args.seed)
# print('Input size =',inputsize,'\nTask info =',taskcla)

# # Inits
# print('Inits...')
# if args.approach == "hat_mask":
#     net=network.Net(inputsize,taskcla,args=args).cuda()
# else:
#     net=network.Net(inputsize,taskcla).cuda()


# utils.print_model_report(net)


# Hyper parameters to search
lamb_values=[0.5, 0.75, 1, 4]          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
smax_values=[200, 400, 800]          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
nepochs_values = [1]
lr_values = [0.003,0.0002,0.05]
optimizer_values = ["adam","sgd"]

# # Hyper parameters to search
# lamb_values=[0.5, 0.75, 1, 4]          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
# smax_values=[200, 400, 800]          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
# nepochs_values = [200]
# lr_values = [0.003,0.0002]
# optimizer_values = ["adam"]

# # Hyper parameters to search
# lamb_values=[0.5, 1]          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
# smax_values=[400]          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
# nepochs_values = [200]
# lr_values = [0.0002]
# optimizer_values = ["adam"]
# # Hyper parameters to search
# lamb_values=[0.75]          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
# smax_values=[400]          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
# nepochs_values = [5]
# lr_values = [0.002]
# optimizer_values = ["sgd"]

# Number of random combinations to try
num_random_combinations = 10

# Generate all combinations
all_combinations = list(itertools.product(lamb_values, smax_values, nepochs_values, lr_values, optimizer_values))

# Randomly sample from all combinations
if num_random_combinations < len(all_combinations):
    random_combinations = random.sample(all_combinations, num_random_combinations)
else:
    random_combinations = all_combinations

# already_exist = [(0.5,800,200,0.003,"adam"),
#  (4,200,200,0.003,"adam"),
#  (0.5,400,200,0.003,"adam")]
already_exist = []


random_combinations = [item for item in random_combinations if item not in already_exist]

exp_folder = args.output
# Loop over hyperparameters
for lamb_val, smax_val, nepochs_val, lr_val, optimizer in random_combinations:
    # Update hyperparameter values
    args.lamb = lamb_val
    args.smax = smax_val
    args.nepochs = nepochs_val
    args.lr = lr_val
    args.optimizer = optimizer

    output_file_prefix = f"{args.datatype}_{args.approach}_{args.experiment}_{args.optimizer}_{args.weight_initializer}_{args.lr}_{args.nepochs}_{args.lamb}_{args.smax}"
    if args.output=='':
        args.output = f"./res/{output_file_prefix}/{output_file_prefix}.txt"
    else:
        # if not os.path.exists(args.)
        Path(f"{exp_folder}/{args.approach}/{output_file_prefix}/").mkdir(parents=True, exist_ok=True)
        args.output = f"{exp_folder}/{args.approach}/{output_file_prefix}/{output_file_prefix}.txt"

    print(f'Hyperparameters: lamb={lamb_val}, smax={smax_val}, nepochs={nepochs_val}, lr={lr_val}')

    ########################################################################################################################
    # Load
    print('Load data...')
    data,taskcla,inputsize=dataloader.get(seed=args.seed)
    print('Input size =',inputsize,'\nTask info =',taskcla)

    # Inits
    print('Inits...')

    if args.approach == "hat_mask":
        from networks import mask as network
        net=network.Net(inputsize,taskcla,args=args).cuda()
    else:
        net=network.Net(inputsize,taskcla).cuda()


    utils.print_model_report(net)

    if args.approach == "hat_mask" or args.approach == "hat":
        appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr,args=args,lamb=args.lamb,smax=args.smax)
    
    elif args.approach == "ewc":
        appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr, lr_min=1e-7,args=args,lamb=args.lamb)

    elif args.approach == "sgd" or args.approach=="pathnet":
        appr=approach.Appr(net,nepochs=args.nepochs,lr=args.lr, lr_min=1e-7,args=args)

    print(appr.criterion)
    utils.print_optimizer_config(appr.optimizer)
    print('-'*100)

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

        # Train
        appr.train(task,xtrain,ytrain,xvalid,yvalid)
        print('-'*100)

        # Test
        for u in range(t+1):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            test_loss,test_acc, test_avg_reg, (target,prediction) = appr.eval(u,xtest,ytest,return_pred=True)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc
            lss[t,u]=test_loss


            test_loss_file_name = "test_loss_"+Path(args.output).stem
            test_loss_file_path = f"{Path(args.output).parent}/{test_loss_file_name}.csv"

            test_prediction_file_name = "test_prediction_"+Path(args.output).stem
            test_prediction_file_path = f"{Path(args.output).parent}/{test_prediction_file_name}.csv"
            # loss_file_path = f"./res/wt_init_{self.args.weight_initializer}_losses_app_{self.args.approach}_exp_{self.args.experiment}_opt_{self.args.optimizer}_lr_{self.lr}_nepoch_{self.nepochs}.csv"
            if os.path.exists(test_loss_file_path):
                test_loss_df = pd.read_csv(test_loss_file_path)
            else:
                test_loss_df = pd.DataFrame(columns=["Task","Test_loss","Test_acc","Avg_test_reg"])

            if os.path.exists(test_prediction_file_path):
                test_pred_df = pd.read_csv(test_prediction_file_path)
            else:
                test_pred_df = pd.DataFrame(columns=["Task","Test_prediction","Test_target"])

            test_loss_df = pd.concat([test_loss_df,pd.DataFrame({"Task":f"{u} {data[u]['name']}",
                                "Test_loss":round(test_loss,6),
                                "Test_acc":round(100*test_acc,6),
                                "Avg_test_reg":round(test_avg_reg,6),
                                # "Val_loss":round(valid_loss,6),
                                # "Val_acc":round(100*valid_acc,6),
                                # "Avg_val_reg":round(valid_avg_reg,6),
                                },index=[0])])

            test_pred_df = pd.concat([test_pred_df,pd.DataFrame({"Task":f"{u} {data[u]['name']}",
                    "Test_target":target,
                    "Test_prediction":prediction,
                    })], ignore_index=True)
        test_loss_df.to_csv(test_loss_file_path,index=False)
        test_pred_df.to_csv(test_prediction_file_path,index=False)

        
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
