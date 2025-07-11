# Skin-Lesion Classification with Pre-trained CNNs

**Paper:** “Improving Skin Lesion Classification with Pre-trained Deep Learning Models”  
IEEE ISBN 979-8-3315-6655-5 (SIU 2025)

## What’s inside
| Folder | Purpose |
|--------|---------|
| `train_skin_cancer_models.py` | Clean end-to-end training script (freezes ImageNet backbones, adds custom head, trains & saves models). |
| `models/` | Saved `.keras` weights after training. |
| `paper/` | PDF of the published paper and BibTeX citation. |

## Quick start
```bash
conda env create -f environment.yml
conda activate skin-cancer-cnn
python train_skin_cancer_models.py

- **Dataset:** ISIC 2019 dermoscopic images  
- **Models:** MobileNet, EfficientNetB7/V2L, VGG16, DenseNet201, ConvNeXt-Base  
- **Task:** Binary classification (Benign vs Malignant)
