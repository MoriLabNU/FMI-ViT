## 📄 paper paper
This repository contains the official implementation of our paper:
**Domain-Specific Pretraining and Fine-Tuning with Contrastive Learning for Fluorescence Microscopic Image Segmentation**  

<p align="center">
  <img src="assets/FMI.png" alt="Results" width="80%">
</p>

## 🚀 Highlights
- **Domain-specific pretraining**: Vision Transformer pretrained on fluorescence microscopy images.  
- **Cross-image foreground-background contrastive learning**: Improves semantic boundary recognition and cross-dataset generalization.  
- **State-of-the-art performance**: Significant IoU and Dice gains over baselines, including on unseen biomarkers.  

## 📂 Repository Structure
```
├── configs/ # Configuration files for training & evaluation
├── datasets/ # Dataset preparation scripts
├── models/ # Model architecture (ViT backbone + segmentation head)
├── weights/ # Pretrained weights
├── utils/ # Helper functions (training, evaluation, visualization)
├── train.py # Training script
├── evaluate.py # Evaluation script
└── README.md
```

## 📊 Dataset Preparation
Prepare fluorescence microscopy datasets as described in the paper.

- **Dataset A**: [Download Link / Preparation Instructions](link_to_dataset_A)  
- **Dataset B**: [Download Link / Preparation Instructions](link_to_dataset_B)

## 💻 Training

### **1. Pretraining (Domain-specific Self-supervised Learning)**
```
python train.py --config configs/pretrain.yaml
```

### **2. Fine-tuning (Foreground-Background Contrastive Learning)**
```
python train.py --config configs/fine_tune.yaml --pretrained weights/pretrained_vit.pth
```

### **3. Evaluation**
```
python train.py --config configs/fine_tune.yaml --pretrained weights/pretrained_vit.pth
```

## 📥 Pretrained Weights
Pretrained weights and fine-tuned models can be downloaded here:
Pretrained ViT (Domain-specific)
Fine-tuned Model (Contrastive Learning)

## 📜 Citation
If you use this repository or our pretrained weights, please cite:
```
@inproceedings{yourbibkey2025,
  title={Domain-Specific Pretraining and Fine-Tuning with Contrastive Learning for Fluorescence Microscopic Image Segmentation},
  author={Your Name and Others},
  booktitle={Proceedings of ...},
  year={2025}
}
```
## 🙏 Acknowledgements
This repository is built upon the excellent works of:

- [DINO](https://github.com/facebookresearch/dino) — Pretraining
- [MMsegmentation](https://github.com/open-mmlab/mmsegmentation) — Fine-tuning

We sincerely thank the authors for releasing their codes and making this research possible.
