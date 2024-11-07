# DFM-trans
Here is the official implementation of the model DFM-trans.

## Abstract
In the era of large-scale pretrained models, efficiently transferring knowledge from source models to new target domains has become a critical research challenge. According to the Multi-source-free Domain Adaptation (MSFDA) issue, the key challenge lies in adapting knowledge from multiple source models to an unlabeled target domain without access to source domain data. However, existing methods typically require extensive parameter tuning for each source model, leading to prohibitive computational costs. To address this, we propose a novel Dynamic Fusion Meta Transformer (DFM-Trans) model that eliminates the need for parameter tuning in the source model backbone. Our innovation lies in the precise integration of knowledge from multiple source domains via an adaptive Feature Fusion Network (FFN) and a dynamic weight adjustment mechanism based on prediction uncertainty. Additionally, we introduce a Comprehensive Confidence-driven Pseudo Label (CCPL) strategy that significantly enhances the model’s adaptability and robustness to the target domain. Extensive experiments on multiple benchmark datasets show that the proposed DFM-Trans not only maintains source domain accuracy but also outperforms existing state-of-the-art methods in terms of target domain adaptation.

## Method
![F1](https://github.com/chengnan1430/DFM-trans/blob/main/image/F1.png)

* First, DFM-Trans model integrates model knowledge from diverse source domains through an adaptive Feature Fusion Network (FFN), combining local and global features to enhance the model's capacity for feature representation.

* Second, DFM-Trans model introduces a dynamic weight adjustment mechanism based on predictive uncertainty, allowing adaptive adjustment of source model weights to optimize performance in the target domain.

* Finally, a comprehensive confidence-driven pseudo-labeling strategy is proposed, prioritizing knowledge extraction from high-confidence samples and transferring it to lower-confidence samples, effectively reducing the generation of erroneous pseudo labels.

## Setup
### Install Package Dependencies

```
* Python Environment: >= 3.6
* torch >= 1.1.0
* torchvision >= 0.3.0
* scipy == 1.3.1
* sklearn == 0.5.0
* numpy == 1.17.4
* argparse, PIL
```

## Datasets:
* **Office Dataset:** Download the datasets [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar) .
* **DomainNet Dataset:** Download [DomainNet](http://ai.bu.edu/DomainNet/) .
* Place these datasets in './data'.
* Using readfile.py to generate '.txt' file for each dataset (change dataset argument in the file accordingly).
data
│ 
└───Office-home
│ │ Art
│ | Clipart
| | Product
| | Real_world 
│ │ Art_list.txt
│ │ Clipart_list.txt
│ │ Product_list.txt
│ │ Real_world_list.txt
└─── DomainNet
│ │ Clipart
│ │ Infograph
│ │   ...
└───  Office-Caltech-
│ │   ...
└───   Office-31
| |   ...

## Training:

* Train source models (shown here for Office-31 with source A)

```shell
python train_source.py --dset office-31 --s 1 --t 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```

* Adapt to target domain (shown here for Office-31 with target D)
```shell
python train_target.py --dset office-31 --t 1 --max_epoch 15 --gpu_id 0 --cls_par 0.7 --crc_par 0.01 --crc_mse 0.01 --output_src ckps/source/ --output ckps/DFM
```










