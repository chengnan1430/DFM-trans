# DFM-trans
Here is the official implementation of the model DFM-trans.

## Abstract
In the era of large-scale pretrained models, efficiently transferring knowledge from multiple sources to new target domains remains a key challenge. In Multi-Source-Free Domain Adaptation (MSFDA), the main difficulty is adapting knowledge from several source models to an unlabeled target domain without access to the original source data. To address this challenge, we introduce the Dynamic Fusion Meta Transformer (DFM-Trans), which integrates source knowledge through an adaptive Feature Fusion Network (FFN) and Multi-Source Uncertainty-Aware Dynamic Weight Strategy. The FFN enriches representations by capturing more expressive features, while the dynamic weighting strategy enables fine-grained, instance-specific model fusion based on prediction uncertainty, without requiring access to or fine-tuning of the source models. In addition, we introduce a Comprehensive Confidence-driven Pseudo Label (CCPL) strategy that leverages dual confidence metrics to filter and refine pseudo labels, significantly enhancing robustness and adaptation effectiveness. Extensive experiments on multiple benchmark datasets demonstrate that the proposed DFM-Trans not only preserves source domain accuracy but also outperforms existing state-of-the-art methods in terms of target domain adaptation.

## Method
![F1](https://github.com/chengnan1430/DFM-trans/blob/main/image/F2.png)

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
  
```
data
│       
└───Office-home
│   │  Art
│   │  Clipart
│   │  Product
|   |  Real_world
|   |  Art_list.txt 
|   |  Clipart_list.txt
|   |  Product_list.txt 
|   |  Real_world_list.txt
└───DomainNet
│   │    Clipart
│   │    Infograph
│   │   ...
└───Office-Caltech
│   │   ...
└───Office-31
│   │   ...
...
```

## Training:

* Train source models (shown here for Office-31 with source A)

```shell
python train_source.py --dset office-31 --s 1 --t 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```

* Adapt to target domain (shown here for Office-31 with target D)
```shell
python train_target.py --dset office-31 --t 1 --max_epoch 15 --gpu_id 0 --cls_par 0.7 --crc_par 0.01 --crc_mse 0.01 --output_src ckps/source/ --output ckps/DFM
```










