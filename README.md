# Adversarial Domain Adaptation with Domain Mixup

<p align="center">
  <img src="docs/model.png" /> 
  <br> The framework of the proposed DM-ADA method.
</p>

<br>

This is the implementation of Adversarial Domain Adaptation with Domain Mixup in PyTorch. This work is accepted as Oral presentation at AAAI 2020.

#### Adversarial Domain Adaptation with Domain Mixup: [[Paper (arxiv)]](https://arxiv.org/abs/1912.01805).
<br>

## Getting Started

* We combine Domain Mixup strategy with a classical adversarial domain adaptation method, [RevGrad](https://arxiv.org/abs/1409.7495v2), to showcase its effectiveness on boosting feature alignment. Details are presented in the [Mixup_RevGrad](https://github.com/ChrisAllenMing/Mixup_for_UDA/Mixup_RevGrad) folder.
* The proposed DM-ADA approach utilizes a VAE-GAN based framework and performs Domain Mixup on both pixel and feature level. Details are presented in the [DM-ADA](https://github.com/ChrisAllenMing/Mixup_for_UDA/DM-UDA) folder.

<p align="center">
  <img src="docs/visda_results.png" width="400" />
</p>

## Citation

If this work helps your research, please cite the following paper (This will be updated when the AAAI paper is publicized).
```
@article{xu2019adversarial,
  title={Adversarial Domain Adaptation with Domain Mixup},
  author={Xu, Minghao and Zhang, Jian and Ni, Bingbing and Li, Teng and Wang, Chengjie and Tian, Qi and Zhang, Wenjun},
  journal={arXiv preprint arXiv:1912.01805},
  year={2019}
}
```