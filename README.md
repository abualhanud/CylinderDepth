# CylinderDepth
CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation

[Paper](https://arxiv.org/abs/2511.16428) | [Supplementary Material](https://arxiv.org/src/2511.16428v2/anc/CylinderDepth_supp.pdf)

## Introduction

Self-supervised surround-view depth estimation enables dense, low-cost 3D perception with a 360° field of view from multiple minimally overlapping images. Yet, most existing methods suffer from depth estimates that are inconsistent across overlapping images. To address this limitation, we propose a novel geometry-guided method for calibrated, time-synchronized multi-camera rigs that predicts dense metric depth. Our approach targets two main sources of inconsistency: the limited receptive field in border regions of single-image depth estimation, and the difficulty of correspondence matching. We mitigate these two issues by extending the receptive field across views and restricting cross-view attention to a small neighborhood. To this end, we establish the neighborhood relationships between images by mapping the image-specific feature positions onto a shared cylinder. Based on the cylindrical positions, we apply an explicit spatial attention mechanism, with non-learned weighting, that aggregates features across images according to their distances on the cylinder. The modulated features are then decoded into a depth map for each view. Evaluated on the DDAD and nuScenes datasets, our method improves both cross-view depth consistency and overall depth accuracy compared with state-of-the-art approaches.

## Code coming soon
