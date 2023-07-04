# Understanding and Benchmarking Zero-Shot Adversarial Robustness for Foundation Models (ICLR 2023)

<p align="center">
  <p align="center" margin-bottom="0px">
    <a href="http://www.cs.columbia.edu/~mcz/"><strong>Chengzhi Mao*</strong></a>
    ·
    <a href=""><strong>Lingyu Zhang</strong></a>
    ·
    <a href=""><strong>Abhishek Joshi</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~junfeng/"><strong>Junfeng Yang</strong></a>
    ·
    <a href=""><strong>Hao Wang</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~vondrick/"><strong>Carl Vondrick</strong></a></p>
    <p align="center" margin-top="0px"><a href="https://arxiv.org/pdf/2212.06079.pdf">https://arxiv.org/pdf/2212.06079.pdf</a></p>
</p>


Deep networks for computer vision are not reliable when they encounter adversarial examples. In this paper, we introduce a framework that uses the dense intrinsic constraints in natural images to robustify inference. By introducing constraints at inference time, we can shift the burden of robustness from training to the inference algorithm, thereby allowing the model to adjust dynamically to each individual image's unique and potentially novel characteristics at inference time. Among different constraints, we find that equivariance-based constraints are most effective, because they allow dense constraints in the feature space without overly constraining the representation at a fine-grained level. Our theoretical results validate the importance of having such dense constraints at inference time. Our empirical experiments show that restoring feature equivariance at inference time defends against worst-case adversarial perturbations. The method obtains improved adversarial robustness on four datasets (ImageNet, Cityscapes, PASCAL VOC, and MS-COCO) on image recognition, semantic segmentation, and instance segmentation tasks. 

# Experiemnt

## Environment

We use anaconda to manage the environment. The configuration file is: ``

## Cityscapes Segmentation Experiment

Download the adversarial pretrained cityscapes checkpoint <a  href="https://cv.cs.columbia.edu/mcz/Equi4Robust/advtrain_drn_d_22_cityscapes.pth.tar">here </a>. 

Download the vanilla pretrained cityscapes checkpoint <a  href="https://cv.cs.columbia.edu/mcz/Equi4Robust/clean_drn_d_22_cityscapes.pth.tar">here </a>. 

`CUDA_VISIBLE_DEVICES=0,1 python equi4robust.py`
