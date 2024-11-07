# DFM-trans
Here is the official implementation of the model DFM-trans.

## Abstract
In the era of large-scale pretrained models, efficiently transferring knowledge from source models to new target domains has become a critical research challenge. According to the Multi-source-free Domain Adaptation (MSFDA) issue, the key challenge lies in adapting knowledge from multiple source models to an unlabeled target domain without access to source domain data. However, existing methods typically require extensive parameter tuning for each source model, leading to prohibitive computational costs. To address this, we propose a novel Dynamic Fusion Meta Transformer (DFM-Trans) model that eliminates the need for parameter tuning in the source model backbone. Our innovation lies in the precise integration of knowledge from multiple source domains via an adaptive Feature Fusion Network (FFN) and a dynamic weight adjustment mechanism based on prediction uncertainty. Additionally, we introduce a Comprehensive Confidence-driven Pseudo Label (CCPL) strategy that significantly enhances the model’s adaptability and robustness to the target domain. Extensive experiments on multiple benchmark datasets show that the proposed DFM-Trans not only maintains source domain accuracy but also outperforms existing state-of-the-art methods in terms of target domain adaptation.

## Method
![F2](https://github.com/chengnan1430/DFM-trans/image/F2.pdf)
