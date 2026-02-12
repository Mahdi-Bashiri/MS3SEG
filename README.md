# MS3SEG: A Multiple Sclerosis MRI Dataset with Tri-Mask Annotations for Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Data-blue.svg)](https://doi.org/10.6084/m9.figshare.30393475)
[![Dataset](https://img.shields.io/badge/Dataset-Figshare-blue.svg)](https://doi.org/10.6084/m9.figshare.30393475)
[![Models](https://img.shields.io/badge/🤗%20Models-Hugging%20Face-yellow.svg)](https://huggingface.co/Bawil/MS3SEG/tree/main)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://www.tensorflow.org/)

Official repository for the MS3SEG dataset presented in our *Scientific Data* paper.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset Access](#dataset-access)
- [Pre-trained Models](#pre-trained-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Usage Guide](#usage-guide)
- [Experimental Scenarios](#experimental-scenarios)
- [Baseline Results](#baseline-results)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🎯 Overview

MS3SEG is a novel MRI dataset comprising **100 multiple sclerosis (MS) patients** from an Iranian cohort with unique **tri-mask annotations** that address a critical clinical challenge: distinguishing pathological MS lesions from normal age-related white matter hyperintensities.

### The Clinical Problem

Not all white matter hyperintensities (WMH) visible on FLAIR imaging represent pathological MS lesions. Age-related changes and CSF contamination artifacts can mimic MS lesions, leading to false-positive segmentations. Our tri-mask annotation framework explicitly addresses this by distinguishing:

1. **Ventricles** - CSF-filled anatomical spaces providing spatial context
2. **Normal WMH** - Age-related hyperintensities or CSF-contaminated regions  
3. **Abnormal WMH** - True MS demyelinating lesions requiring clinical monitoring

### Why MS3SEG?

- **Geographic Diversity**: First large-scale MS dataset from Middle East/North Africa (MENA) region
- **Scanner Diversity**: Toshiba/Canon 1.5T platform (vs. dominant GE/Siemens in existing datasets)
- **Clinical Relevance**: Tri-mask framework enables development of algorithms that make clinically meaningful distinctions
- **Real-World Data**: 2D sequential acquisition reflecting actual clinical practice (not research-only 3D protocols)

---

## ✨ Key Features

### Dataset Characteristics

- ✅ **100 MS patients** (74 female, 26 male; age range: 18-55 years)
- ✅ **Iranian cohort** from Tabriz Medical Center (2018-2022)
- ✅ **Toshiba Vantage 1.5T scanner** (addressing vendor diversity gap)
- ✅ **Multi-sequence MRI**: T1-weighted, T2-weighted, T2-FLAIR (axial + sagittal)
- ✅ **~2000 annotated slices** with expert consensus review
- ✅ **Tri-mask annotations**: Background, Ventricles, Normal WMH, Abnormal WMH
- ✅ **Statistical quality control** to minimize annotation errors

### Data Formats

- Raw DICOM files (~4GB)
- Preprocessed NIfTI volumes (~2GB) - co-registered, standardized (256×256), brain-extracted
- Ground truth tri-mask annotations
- RGB-coded visualization overlays
- Patient metadata (anonymized)
- Cross-validation splits (5-fold + held-out test set)

### Baseline Models

Pre-trained implementations of four state-of-the-art architectures:
- **U-Net** - Classic encoder-decoder with skip connections
- **U-Net++** - Nested skip pathways for reduced semantic gap
- **UNETR** - Transformer-based encoder with CNN decoder
- **Swin UNETR** - Hierarchical shifted-window attention

---

## 📊 Dataset Access

### Complete Dataset

**Primary Repository:** [![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.30393475-blue.svg)](https://doi.org/10.6084/m9.figshare.30393475)

The complete MS3SEG dataset is publicly available at **Figshare** under CC-BY-4.0 license:

🔗 **https://doi.org/10.6084/m9.figshare.30393475**

### Dataset Contents

```
MS3SEG_Dataset/
├── raw_dicom/              # Original DICOM files (4GB)
│   ├── patient_001/
│   │   ├── T1_weighted/
│   │   ├── T2_weighted/
│   │   ├── FLAIR_axial/
│   │   └── FLAIR_sagittal/
│   └── ...
│
├── preprocessed_nifti/     # Preprocessed NIfTI volumes (2GB)
│   ├── patient_001/
│   │   ├── 001_T1_preprocessed.nii.gz
│   │   ├── 001_T2_preprocessed.nii.gz
│   │   └── 001_FLAIR_preprocessed.nii.gz
│   └── ...
│
├── masks/                  # Tri-mask annotations (12-15KB per patient)
│   ├── patient_001/
│   │   └── 001_mask.nii.gz  # Classes: 0=bg, 1=ventricles, 2=normal WMH, 3=abnormal WMH
│   └── ...
│
├── visualizations/         # RGB overlay PNGs (40-60KB per patient)
│   └── patient_001_overlay.png
│
└── metadata/
    ├── patient_demographics.json
    ├── acquisition_parameters.csv
    └── README.txt
```

### Annotation Classes

| Class | Label | Color Code | Description | Clinical Significance |
|-------|-------|------------|-------------|---------------------|
| 0 | Background | Black | Normal brain parenchyma | None |
| 1 | Ventricles | Blue | CSF-filled spaces | Spatial context for lesion localization; source of artifacts |
| 2 | Normal WMH | Green | Age-related hyperintensities or CSF contamination | Should be excluded from MS lesion burden calculations |
| 3 | Abnormal WMH | Red | MS demyelinating lesions | True disease markers requiring clinical monitoring |

---

## 🤗 Pre-trained Models

### Hugging Face Model Hub

Pre-trained weights for all baseline models are available on Hugging Face:

🔗 **https://huggingface.co/YOUR_USERNAME/MS3SEG-models**

### Available Models

| Model | Scenario | Dice Score | Download | Size |
|-------|----------|------------|----------|------|
| U-Net | Multi-class (4-class) | 0.7163 | [u-net-multiclass](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/u-net-multiclass.h5) | ~31MB |
| U-Net | Binary Lesion | 0.7469 | [u-net-binary-lesion](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/u-net-binary-lesion.h5) | ~31MB |
| U-Net | Binary Ventricle | 0.8982 | [u-net-binary-ventricle](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/u-net-binary-ventricle.h5) | ~31MB |
| U-Net++ | Multi-class (4-class) | 0.7094 | [unet++-multiclass](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/unetpp-multiclass.h5) | ~37MB |
| UNETR | Multi-class (4-class) | 0.6136 | [unetr-multiclass](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/unetr-multiclass.h5) | ~90MB |
| Swin UNETR | Multi-class (4-class) | 0.6563 | [swin-unetr-multiclass](https://huggingface.co/YOUR_USERNAME/MS3SEG-models/blob/main/swin-unetr-multiclass.h5) | ~120MB |

### Quick Model Usage

```python
from tensorflow import keras
from huggingface_hub import hf_hub_download

# Download pre-trained model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/MS3SEG-models",
    filename="u-net-multiclass.h5"
)

# Load model
model = keras.models.load_model(model_path, compile=False)

# Use for inference
predictions = model.predict(your_data)
```

---

## 🚀 Installation

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 12GB+ VRAM for transformer models)
- 16GB+ RAM
- ~10GB disk space for repository + data

### Prerequisites

For preprocessing (optional):
- FSL 6.0+ (for brain extraction and co-registration)
- dcm2niix (for DICOM to NIfTI conversion)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Mahdi-Bashiri/MS3SEG.git
cd MS3SEG
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset**

Visit [Figshare](https://doi.org/10.6084/m9.figshare.30393475) and download the dataset. Extract to a `data/` directory:

```bash
mkdir data
# Extract downloaded files to data/
```

5. **Verify installation**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
```

---

## ⚡ Quick Start

### 1. Using Pre-trained Models

```python
from models import build_unet
from utils import MS3SEGDataLoader, calculate_multiclass_metrics
import numpy as np

# Load data
data_loader = MS3SEGDataLoader(data_dir="data/preprocessed_nifti")
images, masks = data_loader.load_dataset(
    patient_ids=['001', '002', '003'], 
    scenario='multi_class'
)

# Load pre-trained model (or download from Hugging Face)
from tensorflow import keras
model = keras.models.load_model('path/to/pretrained_model.h5', compile=False)

# Make predictions
predictions = model.predict(images, batch_size=4)
pred_classes = np.argmax(predictions, axis=-1)

# Evaluate
metrics = calculate_multiclass_metrics(
    masks, pred_classes, 
    num_classes=4,
    class_names=['Background', 'Ventricles', 'Normal WMH', 'Abnormal WMH']
)
print(metrics)
```

### 2. Training a New Model

```bash
# Train U-Net on multi-class segmentation (all folds)
python training/train.py --scenario multi_class --models U-Net

# Train specific fold
python training/train.py --scenario multi_class --models U-Net --fold 1

# Train all models on binary lesion segmentation
python training/train.py --scenario binary_abnormal_wmh

# Custom configuration
python training/train.py --config my_config.json --scenario three_class
```

### 3. Evaluating Models

```bash
# Evaluate all models for a scenario
python training/evaluate.py \
    --results_dir results/experiment_20250212_120000 \
    --scenario multi_class

# Evaluate with visualizations
python training/evaluate.py \
    --results_dir results/experiment_20250212_120000 \
    --scenario multi_class \
    --visualize
```

### 4. Creating Visualizations

```python
from utils import MS3SEGVisualizer
import nibabel as nib

# Initialize visualizer
viz = MS3SEGVisualizer(save_dir='figures')

# Load sample data
flair = nib.load('data/patient_001/001_FLAIR_preprocessed.nii.gz').get_fdata()
mask = nib.load('data/masks/patient_001/001_mask.nii.gz').get_fdata()

# Create publication-quality figure
slice_idx = 10
viz.plot_sample_with_mask(
    flair[:, :, slice_idx],
    mask[:, :, slice_idx],
    title='MS Patient with Tri-Mask Annotation',
    save_name='figure_sample'
)
```

---

## 📁 Repository Structure

```
MS3SEG/
├── preprocessing/          # Preprocessing pipeline scripts
│   ├── anonymization.py          # Remove patient identifiers from DICOM
│   ├── dicom_to_nifti.py         # Convert DICOM to NIfTI format
│   ├── coregistration_and_brain_extraction.py  # FLIRT + BET
│   └── standardization.py        # Resize to 256×256
│
├── models/                 # Deep learning model architectures
│   ├── __init__.py              # Model registry and factory
│   ├── unet.py                  # U-Net implementation
│   ├── unet_plusplus.py         # U-Net++ with nested skip pathways
│   ├── unetr.py                 # UNETR (Vision Transformer encoder)
│   └── swin_unetr.py            # Swin UNETR (Hierarchical attention)
│
├── training/              # Training and evaluation orchestration
│   ├── train.py                 # Main training script (k-fold CV)
│   ├── evaluate.py              # Comprehensive evaluation
│   └── config.json              # Hyperparameters and settings
│
├── utils/                 # Utility functions and classes
│   ├── __init__.py
│   ├── data_loader.py           # MS3SEGDataLoader class
│   ├── metrics.py               # Dice, IoU, HD95 + loss functions
│   └── visualization.py         # MS3SEGVisualizer class
│
├── splits/                # Cross-validation splits
│   ├── 5fold_splits.json        # 5-fold CV patient assignments
│   └── test_set.json            # Held-out test patients
│
├── figures/               # Generated figures (paper + supplementary)
├── tables/                # Results tables (CSV format)
│
├── requirements.txt       # Python package dependencies
├── LICENSE               # MIT License for code
├── README.md             # This file
└── SETUP_GUIDE.md        # Detailed setup and usage instructions
```

---

## 📖 Usage Guide

### Configuration

All training parameters are controlled via `training/config.json`:

```json
{
  "data": {
    "data_dir": "data/preprocessed_nifti",
    "target_size": [256, 256]
  },
  "training": {
    "epochs": 100,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "loss_function": "unified_focal_loss"
  },
  "models": {
    "to_compare": ["U-Net", "UNet++", "UNETR", "SwinUNETR"]
  }
}
```

### Preprocessing Pipeline

If starting from raw DICOM files:

```bash
# 1. Anonymize DICOM headers
python preprocessing/anonymization.py \
    --input raw_dicom/ \
    --output anonymized_dicom/

# 2. Convert to NIfTI
python preprocessing/dicom_to_nifti.py \
    --input anonymized_dicom/ \
    --output nifti/

# 3. Co-register sequences to FLAIR space
python preprocessing/coregistration_and_brain_extraction.py \
    --t1 nifti/patient_001_T1.nii.gz \
    --t2 nifti/patient_001_T2.nii.gz \
    --flair nifti/patient_001_FLAIR.nii.gz \
    --output registered/

# 4. Standardize dimensions
python preprocessing/standardization.py \
    --input registered/ \
    --output preprocessed/ \
    --target_size 256 256
```

### Advanced Training Options

```python
from training.train import MS3SEGTrainer

# Initialize trainer
trainer = MS3SEGTrainer(config_path='training/config.json')

# Customize for specific research question
trainer.config['training']['loss_function'] = 'dice'
trainer.config['training']['batch_size'] = 8

# Run experiment
results = trainer.run_kfold_experiment(
    scenario='multi_class',
    models=['U-Net', 'UNet++']
)
```

### Creating Custom Scenarios

```python
# Add to utils/data_loader.py
def transform_masks_for_scenario(masks, scenario):
    if scenario == 'my_custom_scenario':
        # Your custom transformation
        transformed = masks.copy()
        # ... modifications ...
        return transformed
```

### Using Individual Components

```python
# Just need metrics?
from utils.metrics import dice_coefficient, hausdorff_distance_95

dice = dice_coefficient(y_true, y_pred)
hd95 = hausdorff_distance_95(y_true, y_pred, voxel_spacing=(0.9, 0.9))

# Just need a model?
from models import build_unet, build_unet_plusplus

model = build_unet(input_shape=(256, 256, 1), num_classes=4, filters=64)
# or
model = build_unet_plusplus(input_shape=(256, 256, 1), num_classes=4)

# Just need visualization?
from utils import MS3SEGVisualizer

viz = MS3SEGVisualizer(save_dir='my_figures')
viz.plot_predictions_comparison(image, ground_truth, predictions_dict)
```

---

## 🔬 Experimental Scenarios

The repository supports four segmentation scenarios as described in the paper:

### 1. Multi-Class Tri-Mask Segmentation

**Description:** Simultaneous 4-class segmentation (background, ventricles, normal WMH, abnormal WMH)

**Use Case:** Full exploitation of tri-mask annotations; developing anatomically-aware algorithms

**Command:**
```bash
python training/train.py --scenario multi_class
```

### 2. Three-Class Segmentation

**Description:** 3-class segmentation merging normal WMH into background (background, ventricles, abnormal WMH)

**Use Case:** Focusing on pathological structures while maintaining anatomical context

**Command:**
```bash
python training/train.py --scenario three_class
```

### 3. Binary Lesion Segmentation

**Description:** Binary segmentation of abnormal WMH (MS lesions) vs. everything else

**Use Case:** Traditional MS lesion segmentation; comparison with existing datasets/methods

**Command:**
```bash
python training/train.py --scenario binary_abnormal_wmh
```

### 4. Binary Ventricle Segmentation

**Description:** Binary segmentation of ventricles vs. everything else

**Use Case:** Anatomical structure segmentation; demonstrating dataset utility beyond lesion analysis

**Command:**
```bash
python training/train.py --scenario binary_ventricles
```

---

## 📈 Baseline Results

### Multi-Class Segmentation (5-fold Cross-Validation)

Results reported as mean ± standard deviation across 5 folds:

| Model | Ventricles<br>Dice ↑ | Normal WMH<br>Dice ↑ | Abnormal WMH<br>Dice ↑ | Mean<br>Dice ↑ | HD95 (mm) ↓ |
|-------|:-------------------:|:-------------------:|:---------------------:|:-------------:|:-----------:|
| **U-Net** | **0.8897 ± 0.0049** | **0.5919 ± 0.0070** | **0.6672 ± 0.0177** | **0.7163** | **21.71** |
| U-Net++ | 0.8934 ± 0.0018 | 0.5859 ± 0.0061 | 0.6490 ± 0.0226 | 0.7094 | 20.40 |
| UNETR | 0.8240 ± 0.0136 | 0.4618 ± 0.0191 | 0.5551 ± 0.0486 | 0.6136 | 29.31 |
| Swin UNETR | 0.8632 ± 0.0056 | 0.5164 ± 0.0158 | 0.5893 ± 0.0071 | 0.6563 | 25.05 |

**Key Findings:**
- U-Net achieves best overall performance with consistent results across folds
- Transformer-based models (UNETR, Swin UNETR) underperform, likely due to limited training data (64 patients/fold)
- Normal WMH class is most challenging (DSC 0.46-0.59) due to diffuse, irregular boundaries
- Ventricle segmentation is most reliable (DSC > 0.82 for all models)

### Binary Segmentation Results

#### Lesion Segmentation (Abnormal WMH)

| Model | Dice ↑ | IoU ↑ | HD95 (mm) ↓ |
|-------|:------:|:-----:|:-----------:|
| **U-Net** | **0.7469 ± 0.0060** | **0.5965** | **32.51** |
| U-Net++ | 0.6383 ± 0.0328 | 0.5073 | 30.99 |
| UNETR | 0.6428 ± 0.0137 | 0.4738 | 42.18 |
| Swin UNETR | 0.6088 ± 0.0392 | 0.4374 | 37.76 |

#### Ventricle Segmentation

| Model | Dice ↑ | IoU ↑ | HD95 (mm) ↓ |
|-------|:------:|:-----:|:-----------:|
| **U-Net** | **0.8982 ± 0.0029** | **0.8154** | **9.84** |
| U-Net++ | 0.8862 ± 0.0082 | 0.7960 | 10.33 |
| Swin UNETR | 0.8610 ± 0.0017 | 0.7562 | 12.80 |
| UNETR | 0.8457 ± 0.0062 | 0.7329 | 14.79 |

**Complete results and per-fold breakdowns available in `tables/` directory.**

### Performance Analysis

**Common Challenges:**
- Small lesions (<5mm) frequently under-segmented across all architectures
- Periventricular regions prone to false positives (normal WMH misclassified as lesions)
- Boundary delineation more difficult for irregular MS lesions vs. smooth ventricle surfaces

**Dataset Characteristics:**
- Binary lesion segmentation (DSC 0.75) easier than multi-class abnormal WMH (DSC 0.67)
- Explicit normal vs. abnormal WMH distinction adds ~10% complexity but critical for clinical validity
- Class imbalance: Background 99.55%, Ventricles 0.3%, Normal WMH 0.05%, Lesions 0.1%

---

## 📝 Citation

If you use this dataset, code, or pre-trained models in your research, please cite our paper:

```bibtex
@article{bashiri2025ms3seg,
  title={MS3SEG: A Multiple Sclerosis MRI Dataset with Tri-Mask Annotations for Lesion Segmentation},
  author={Bashiri Bawil, Mahdi and Shamsi, Mousa and Ghalehasadi, Aydin and Jafargholkhanloo, Ali Fahmi and Shakeri Bavil, Abolhassan},
  journal={Scientific Data},
  year={2025},
  volume={XX},
  pages={XXX},
  doi={10.XXXX/XXXXX},
  publisher={Nature Publishing Group}
}
```

**Dataset DOI:** https://doi.org/10.6084/m9.figshare.30393475

---

## 🤝 Contributing

We welcome contributions from the research community! Here's how you can help:

### Reporting Issues

Found a bug or have a feature request?
- Check [existing issues](https://github.com/Mahdi-Bashiri/MS3SEG/issues) first
- Create a [new issue](https://github.com/Mahdi-Bashiri/MS3SEG/issues/new) with:
  - Clear description
  - Steps to reproduce (for bugs)
  - Expected vs. actual behavior
  - System information (OS, Python version, GPU)

### Contributing Code

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas where contributions are especially welcome:**
- Additional model architectures
- Data augmentation strategies
- Post-processing techniques
- Preprocessing improvements
- Documentation enhancements
- Bug fixes

### Contribution Guidelines

- Follow existing code style (PEP 8 for Python)
- Add docstrings to new functions/classes
- Include unit tests for new functionality
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

### Code License

This repository's code is released under the **MIT License**. See [LICENSE](LICENSE) file for details.

```
Copyright (c) 2025 Mahdi Bashiri Bawil, Mousa Shamsi, Aydin Ghalehasadi, 
Ali Fahmi Jafargholkhanloo, Abolhassan Shakeri Bavil
```

### Dataset License

The MS3SEG dataset is released under **Creative Commons Attribution 4.0 International (CC-BY-4.0)** license.

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and cite our paper

See dataset repository for complete license: https://doi.org/10.6084/m9.figshare.30393475

---

## 🙏 Acknowledgments

### Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

### Collaborators & Support

- **Golgasht Medical Imaging Center, Tabriz, Iran** - Data acquisition
- **Eng. Mehrdad Rahbarpour** - Technical assistance in data acquisition
- **Eng. Azad Ahmadi** - Technical assistance in data acquisition
- **All participating patients** - For providing informed consent for data sharing

### Ethics Approval

This study was approved by the **Tabriz University of Medical Sciences Research Ethics Committee** (IR.TBZMED.REC.1402.902). Written informed consent was obtained from all patients.

### Tools & Software

We acknowledge the developers of the following open-source tools used in this work:
- [FSL](https://fsl.fmrib.ox.ac.uk/) - Image preprocessing
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [NiBabel](https://nipy.org/nibabel/) - Neuroimaging file I/O
- [scikit-image](https://scikit-image.org/) - Image processing

### AI Assistance

We acknowledge the use of **Claude Sonnet 4** (Anthropic, accessed October 2024-February 2025) for assistance in manuscript preparation, code documentation, and README creation. All AI-generated content was carefully reviewed, edited, and validated by the authors.

---

## 📧 Contact

### Primary Contact

**Mahdi Bashiri Bawil**
- 📧 Email: mehdi.bashiri.b@gmail.com, m_bashiri99@sut.ac.ir
- 🔗 GitHub: [@Mahdi-Bashiri](https://github.com/Mahdi-Bashiri)
- 🎓 ORCID: [0009-0002-2029-3245](https://orcid.org/0009-0002-2029-3245)

### Corresponding Author

**Dr. Mousa Shamsi**
- 📧 Email: shamsi@sut.ac.ir
- 🏛️ Affiliation: Biomedical Engineering Faculty, Sahand University of Technology, Tabriz, Iran
- 🎓 ORCID: [0000-0003-4670-0531](https://orcid.org/0000-0003-4670-0531)

### Support Channels

- **GitHub Issues:** [Report bugs or request features](https://github.com/Mahdi-Bashiri/MS3SEG/issues)
- **GitHub Discussions:** [Ask questions or discuss ideas](https://github.com/Mahdi-Bashiri/MS3SEG/discussions)
- **Email:** For private inquiries or collaboration proposals

---

## 📚 Related Resources

### Our Publications

- **Main Paper:** [MS3SEG on Scientific Data](https://doi.org/10.6084/m9.figshare.30393475)
- **Dataset:** [MS3SEG on Figshare](https://doi.org/10.6084/m9.figshare.30393475)
- **Models:** [MS3SEG on Hugging Face](https://huggingface.co/YOUR_USERNAME/MS3SEG-models)

### Other MS Datasets

- [ISBI 2015 Longitudinal MS Challenge](https://smart-stats-tools.org/lesion-challenge)
- [MICCAI 2016 MSSEG Challenge](https://portal.fli-iam.irisa.fr/msseg-2/)
- [MSLesSeg Dataset (2025)](https://www.nature.com/articles/s41597-025-05250-y)

### Relevant Papers

1. Filippi et al., "Multiple sclerosis," *Nature Reviews Disease Primers*, 2018
2. Thompson et al., "Diagnosis of Multiple Sclerosis: 2017 Revisions of the McDonald Criteria," *Lancet Neurology*, 2018
3. Carass et al., "Longitudinal multiple sclerosis lesion segmentation," *NeuroImage*, 2017
4. Commowick et al., "Multiple sclerosis lesions segmentation from multiple experts," *NeuroImage*, 2021

---

## 🔄 Updates & Changelog

### Version 1.0.0 (February 2025) - Initial Release
- ✅ Complete dataset (100 patients) released on Figshare
- ✅ Baseline implementations (U-Net, U-Net++, UNETR, Swin UNETR)
- ✅ Pre-trained model weights on Hugging Face
- ✅ 5-fold cross-validation results
- ✅ Comprehensive documentation and tutorials

### Planned Updates
- 🔄 Additional data augmentation strategies
- 🔄 Transfer learning from larger datasets
- 🔄 Ensemble methods evaluation
- 🔄 Longitudinal data collection (future cohort)
- 🔄 3D/isotropic acquisition protocol (in progress)

**Last Updated:** February 12, 2025

---

## ⭐ Star History

If you find this work useful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/MS3SEG&type=Date)](https://star-history.com/#Mahdi-Bashiri/MS3SEG&Date)

---

## 🎯 Keywords

`multiple-sclerosis` `medical-imaging` `mri-segmentation` `deep-learning` `u-net` `transformers` `neuroimaging` `dataset` `brain-segmentation` `lesion-detection` `white-matter-hyperintensities` `tensorflow` `computer-vision` `medical-ai`

---

<div align="center">

**[⬆ Back to Top](#ms3seg-a-multiple-sclerosis-mri-dataset-with-tri-mask-annotations-for-lesion-segmentation)**

Made with ❤️ by the MS3SEG Team

</div>

