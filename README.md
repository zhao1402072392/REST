# REST: Enhancing Group Robustness in DNNs through Reweighted Sparse Training  

[REST: Enhancing Group Robustness in DNNs through Reweighted Sparse Training](link)  
Jiaxu Zhao*, [Lu Yin*](https://luuyin.com/), [SHiwei Liu](https://shiweiliuiiiiiii.github.io/), [Meng Fang](https://mengf1.github.io/), [Mykola Pechenizkiy](https://www.tue.nl/en/research/researchers/mykola-pechenizkiy)  
## Abstract
The deep neural network (DNN) has been proven effective in various domains. However, they often struggle to perform well on certain minority groups during inference, despite showing strong performance on the majority of data groups. This is because over-parameterized models learned bias attributes from a large number of bias-aligned training samples. These bias attributes are strongly spuriously correlated with the target variable, causing the models to be biased towards spurious correlations (i.e., bias-conflicting). To tackle this issue, we propose a novel reweighted sparse training framework, dubbed as REST, which aims to enhance the performance of biased data while improving computation and memory efficiency. Our proposed REST framework has been experimentally validated on three datasets, demonstrating its effectiveness in exploring unbiased subnetworks. We found that REST reduces the reliance on spuriously correlated features, leading to better performance across a wider range of data groups with fewer training and inference resources. We highlight that the REST framework represents a promising approach for improving the performance of DNNs on biased data, while simultaneously improving computation and memory efficiency. By reducing the reliance on spurious correlations, REST has the potential to enhance the robustness of DNNs and improve their generalization capabilities.
## Prerequisites
- python 3.6.8
- matplotlib 3.0.3
- numpy 1.16.2
- pandas 0.24.2
- pillow 5.4.1
- pytorch 1.1.0
- pytorch_transformers 1.2.0
- torchvision 0.5.0a0+19315e3
- tqdm 4.32.2

## Datasets and code 

### Dataset
Download the datasets with the following [URL](https://drive.google.com/drive/folders/1JEOqxrhU_IhkdcRohdbuEtFETUxfNmNT). Note that BFFHQ is the dataset used in "BiaSwap: Removing Dataset Bias with Bias-Tailored Swapping Augmentation" (Kim et al., ICCV 2021). Unzip the files and the directory structures will be as following:
### Scripts
A sample command to train the ResNet18 model on CMNIST is:
`pct=(0.5 1 2 5)
upw=(10 30 50 80)
for  i in 0 1 2 3
do
for density in 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 41 42 43
    do
    python run_expt.py -d cmnist \
        --seed $seed\
        --lr 0.01 --batch_size 256 --weight_decay 0.0 \
      --model CNN --n_epochs 100 --train_from_scratch --sparse \
      --density $density --update_frequency 1000 --conflict_pct ${pct[i]} --lambda_upweight ${upw[i]} --scheduler
    done
done
done
 --scheduler
    `. 
    
A sample command to train the ResNet18 model on Cifar10c is:
`pct=(0.5 1 2 5)
upw=(10 30 50 80)
cd ..
for  i in 0 1 2 3
do
for density in 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 41 42 43
    do
    python run_expt.py -d cifar10c \
        --seed $seed\
        --lr 0.001 --batch_size 256 --weight_decay 0.0 \
      --model resnet18vw --n_epochs 100 --train_from_scratch --resnet_width 64 --sparse \
      --density $density --update_frequency 1000  --conflict_pct ${pct[i]} --lambda_upweight ${upw[i]} --scheduler
    done
done
done`

A sample command to train the ResNet18 model on BBFHQ is:
`for density in 0.0005 0.001 0.005 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 41 42 43
    do
    python run_expt.py -d bffhq \
        --seed $seed\
        --lr 0.001 --batch_size 256 --weight_decay 0.0 \
      --model resnet18vw --n_epochs 100 --train_from_scratch --resnet_width 64 --sparse \
      --density $density --update_frequency 1000 --conflict_pct 0.5 --lambda_upweight 80 --scheduler
    done
done`
