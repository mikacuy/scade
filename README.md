# SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates 
**[SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates](https://scade-spacecarving-nerfs.github.io)** 

Mikaela Angelina Uy, Ricardo Martin-Brualla, Leonidas Guibas and Ke Li

CVPR 2023


![pic-network](scade_teaser_corrected.png)

## Introduction
We present SCADE, a novel technique for NeRF reconstruction under **sparse, unconstrained views** for **in-the-wild indoor scenes**. We leverage on generalizable monocular depth priors and address to represent the inherent ambiguities of monocular depth by exploiting our **ambiguity-aware depth estimates** (leftmost). Our approach accounts for **multimodality** of both distributions using our novel **space carving loss** that seeks to **disambiguate** and find the common mode to fuse the information between different views (middle). As shown (rightmost), SCADE enables better photometric reconstruction especially in highly ambiguous scenes such as non-opaque surfaces.

```
@inproceedings{uy-scade-cvpr23,
      title = {SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates},
      author = {Mikaela Angelina Uy and Ricardo Martin-Brualla and Leonidas Guibas and Ke Li},
      booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2023}
  }
```

## SCADE Pretrained Models
SCADE pretrained models can be downloaded [here](http://download.cs.stanford.edu/orion/scade/pretrained_models.zip).

## Ambiguity-Aware Prior Pretrained Model
Our Ambiguity-aware prior pretrained model [here](http://download.cs.stanford.edu/orion/scade/ambiguity_aware_prior_pretrained_model.zip).

This model predicts a distribution of possible depth to capture the different ambiguities, such as albedo vs shading (top), different scales of concavity (middle) and multimodality in the outputs of non-opaque surfaces (bottom) as shown below.
![pic-network](ambiguity_aware_prior_estimates.png)

## Datasets
Our processed datasets can be downloaded [here](http://download.cs.stanford.edu/orion/scade/datasets.zip).

## Code
TODO Documentation.

## Miscellaneous
[DDP](https://github.com/barbararoessle/dense_depth_priors_nerf) depth completion prior trained on the Taskonomy dataset can be found [here](http://download.cs.stanford.edu/orion/scade/ddp_completion_taskonomy_prior.zip).
