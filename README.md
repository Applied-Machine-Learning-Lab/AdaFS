# AdaFS: Adaptive Feature Selection in Deep Recommender System

---

Source code of [AdaFS: Adaptive Feature Selection in Deep Recommender System](https://dl.acm.org/doi/abs/10.1145/3534678.3539204), in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

!['Img_AdaFS'](/AdaFS.jpg "AdaFS")

## Abstract

---
Feature selection plays an impactful role in deep recommender systems, which selects a subset of the most predictive features, so as to boost the recommendation performance and accelerate model optimization. The majority of existing feature selection methods, however, aim to select only a fixed subset of features. This setting cannot fit the dynamic and complex environments of practical recommender systems, where the contribution of a specific feature varies significantly across user-item interactions. In this paper, we propose an adaptive feature selection framework, AdaFS, for deep recommender systems. To be specific, we develop a novel controller network to automatically select the most relevant features from the whole feature space, which fits the dynamic recommendation environment better. Besides, different from classic feature selection approaches, the proposed controller can adaptively score each example of user-item interactions, and identify the most informative features correspondingly for subsequent recommendation tasks. We conduct extensive experiments based on two public benchmark datasets from a real-world recommender system. Experimental results demonstrate the effectiveness of AdaFS, and its excellent transferability to the most popular deep recommendation models.

## Training

---
Run the following code to train the model:

``` python
python main.py --model_name [AdaFS_soft/AdaFS_hard] --dataset_name movielens1M 
```

Note that the AdsFS_hard selects 50% features by default. Please feel free to change it with hypereparameter *k*.

## Citation

---
Please cite with the below bibTex if you find it helpful to your research.
```
@inproceedings{lin2022adafs,
  title={AdaFS: Adaptive Feature Selection in Deep Recommender System},
  author={Lin, Weilin and Zhao, Xiangyu and Wang, Yejing and Xu, Tong and Wu, Xian},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3309--3317},
  year={2022}
}
```
