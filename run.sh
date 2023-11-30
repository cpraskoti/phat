# python src/run_independent_task.py --seed 1 --lr 0.0005 --experiment cifar --approach hat --nepochs 500 --optimizer adam --weight_initializer xavier --output ./res/good_exp/indep_hat_500_0.0005_xavier_adam_cifar

# python src/run_independent_task.py --seed 1 --lr 0.0005 --experiment cifar --approach hat_mask --nepochs 500 --optimizer adam --weight_initializer xavier --output ./res/good_exp/indep_hat_mask_500_0.0005_xavier_adam_cifar

# python src/run.py --seed 1 --experiment cifar --approach hat --weight_initializer xavier --output ./res/hyper_param/

# python src/run.py --seed 1 --experiment cifar --approach hat_mask --weight_initializer xavier --output ./res/hyper_param/

# python src/run_cifar_split.py --seed 1 --experiment cifar --datatype cifar10 --approach hat_mask --weight_initializer xavier --output ./res/cifar_split/hyperparam/cifar10/  


 python src/run_cifar_split.py --seed 1 --experiment cifar --datatype cifar100 --approach hat_mask --weight_initializer xavier --output ./res/cifar_split/hyperparam/cifar100/