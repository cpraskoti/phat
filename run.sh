python src/run_independent_task.py --seed 1 --lr 0.0005 --experiment cifar --approach hat --nepochs 500 --optimizer adam --weight_initializer xavier --output ./res/good_exp/indep_hat_500_0.0005_xavier_adam_cifar

python src/run_independent_task.py --seed 1 --lr 0.0005 --experiment cifar --approach hat_mask --nepochs 500 --optimizer adam --weight_initializer xavier --output ./res/good_exp/indep_hat_mask_500_0.0005_xavier_adam_cifar

