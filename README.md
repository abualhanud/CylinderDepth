
# CylinderDepth

  

**CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation**


[Project Page](https://abualhanud.github.io/CylinderDepthPage/) | [Paper](https://arxiv.org/abs/2511.16428) | [Supplementary Material](https://arxiv.org/src/2511.16428v3/anc/CylinderDepth_supp.pdf)


  



  

# Data Preparation



  

To generate the ground-truth depth labels:

```bash
python tools/export_gt_depth_ddad.py
python tools/export_gt_depth_nusc.py
```


To generate the overlap ground-truth depth labels:

```bash
python tools/export_overlap_depth_ddad.py
python tools/export_overlap_depth_nuscenes.py
```


# Checkpoints

  

You can download the pre-trained checkpoints for ddad and nuscens here:
https://huggingface.co/samerabualhanud/CylinderDepth/tree/main
  

# Environment Setup

  
```bash
 cd CylinderDepth
 conda env create -f CylinderDepth.yml
 conda activate CylinderDepth
 pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
 cd external/dgp
 pip install -r requirements.txt -r requirements-dev.txt
 pip install --editable .
```
  

# Training

  

To train the model:

Change the data paths in ```dataset/ddad_dataset.py```,```dataset/nusc_dataset.py``` and the config files.
  
### DDAD
```bash
python  train.py  \

--config_file  ./configs/ddp/nuscenes/nusc_baseline_352_ddp_min_1.0_front_sp_con_0.001_sptp_con_0.05_flipv5_34_106.yaml
```

### nuScenes
```bash
python  train.py  \

--config_file  ./configs/ddp/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml
``` 


# Evaluation

To evaluate the model on the overlap depth, set ```overlap``` in the config file to ```True```.

To evaluate the model:

```bash
python  eval.py  \

--config_file  ./configs/ddp/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5.yaml  \

--weight_path  ./results/baseline_ddp_384_front_sp_con_pre_0.001_sptp_con_0.2_flipv5/models/weights_19
```

# Cite
If you find our work useful, please consider citing our paper:

```bibtex
@article{abualhanud2025cylinderdepth,
  title={CylinderDepth: Cylindrical Spatial Attention for Multi-View Consistent Self-Supervised Surround Depth Estimation},
  author={Abualhanud, Samer and Grannemann, Christian and Mehltretter, Max},
  journal={arXiv preprint arXiv:2511.16428},
  year={2025}
}
```


# Notes
We would like to thank the authors of [VFDepth](https://github.com/42dot/VFDepth), [SurroundDepth](https://github.com/weiyithu/SurroundDepth), [CVCDepth](https://github.com/denyingmxd/CVCDepth), and [MonoDepth2](https://github.com/nianticlabs/monodepth2). This codebase builds upon and benefits greatly from their valuable open-source contributions.