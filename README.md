[ms3seg_readme.md](https://github.com/user-attachments/files/22866208/ms3seg_readme.md)
# MS3SEG: A Multiple Sclerosis MRI Dataset with Tri-Mask Annotations for Lesion Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Data-blue.svg)](PAPER_LINK_HERE)

Official repository for the MS3SEG dataset presented in our *Scientific Data* paper.

## Overview

MS3SEG is a novel MRI dataset comprising 100 multiple sclerosis (MS) patients with unique **tri-mask annotations** that distinguish:
- **Ventricles** (CSF-filled spaces)
- **Normal white matter hyperintensities** (age-related or CSF contamination)
- **Abnormal white matter hyperintensities** (MS lesions)

### Key Features
- 100 MS patients from Iranian cohort
- Acquired on 1.5T Toshiba Vantage scanner
- Multi-sequence MRI: T1-weighted, T2-weighted, T2-FLAIR (axial + sagittal)
- Expert-validated tri-mask annotations with quality control
- Baseline benchmarks: U-Net, U-Net++, UNETR, Swin UNETR

## Dataset Access

The complete MS3SEG dataset is publicly available:

**Dataset Repository:** [DOI/Link to Figshare or Zenodo - TO BE ADDED]

**License:** CC-BY-4.0

### Dataset Contents
- Raw DICOM files
- Preprocessed NIfTI volumes (co-registered, standardized, brain-extracted)
- Ground truth tri-mask annotations
- Visualization overlays
- Patient demographics (anonymized)

## Repository Contents

```
MS3SEG/
├── preprocessing/          # Preprocessing scripts
│   ├── anonymization.py
│   ├── dicom_to_nifti.py
│   ├── coregistration.py
│   ├── standardization.py
│   └── brain_extraction.py
├── models/                 # Model implementations
│   ├── unet.py
│   ├── unet_plusplus.py
│   ├── unetr.py
│   └── swin_unetr.py
├── training/              # Training and evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── config.yaml
├── utils/                 # Utility functions
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py
├── figures/               # Figures from the paper
├── tables/                # Tables from the paper
├── splits/                # Cross-validation splits
│   ├── 5fold_splits.json
│   └── 5fold_splits_anonymized.json
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

## Installation

### Requirements
- Python 3.8+
- TensorFlow 2.x
- FSL 6.0+ (for preprocessing)
- dcm2niix (for DICOM conversion)

### Setup

```bash
# Clone the repository
git clone https://github.com/Mahdi-Bashiri/MS3SEG.git
cd MS3SEG

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing

```bash
# Convert DICOM to NIfTI
python preprocessing/dicom_to_nifti.py --input /path/to/dicom --output /path/to/nifti

# Co-register sequences
python preprocessing/coregistration.py --t1 t1.nii.gz --t2 t2.nii.gz --flair flair.nii.gz --output /path/to/output

# Brain extraction
python preprocessing/brain_extraction.py --input flair.nii.gz --output brain_extracted.nii.gz
```

### 2. Training

```bash
# Train U-Net on multi-class segmentation
python training/train.py --model unet --scenario multiclass --config training/config.yaml

# Train with specific fold
python training/train.py --model unet --scenario multiclass --fold 1
```

### 3. Evaluation

```bash
# Evaluate trained model
python training/evaluate.py --model /path/to/model.h5 --data /path/to/test_data --scenario multiclass
```

## Experimental Scenarios

The code supports three experimental scenarios as described in the paper:

1. **Multi-class Tri-Mask Segmentation**: 4-class segmentation (background, ventricles, normal WMH, abnormal WMH)
2. **Binary Ventricle Segmentation**: Ventricle vs. background
3. **Binary Lesion Segmentation**: Abnormal WMH vs. background

## Baseline Results

| Method | Ventricles DSC | Normal WMH DSC | Abnormal WMH DSC |
|--------|----------------|----------------|------------------|
| U-Net | 0.8897 ± 0.0049 | 0.5919 ± 0.0070 | 0.6672 ± 0.0177 |
| U-Net++ | 0.8934 ± 0.0018 | 0.5859 ± 0.0061 | 0.6490 ± 0.0226 |
| UNETR | 0.8240 ± 0.0136 | 0.4618 ± 0.0191 | 0.5551 ± 0.0486 |
| Swin UNETR | 0.8632 ± 0.0056 | 0.5164 ± 0.0158 | 0.5893 ± 0.0071 |

See `tables/` directory for complete results.

## Citation

If you use this dataset or code, please cite our paper:

```bibtex
@article{ms3seg2025,
  title={MS3SEG: A Multiple Sclerosis MRI Dataset with Tri-Mask Annotations for Lesion Segmentation},
  author={[Author Names]},
  journal={Scientific Data},
  year={2025},
  doi={[DOI_TO_BE_ADDED]}
}
```

## Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## License

- **Code:** MIT License
- **Dataset:** CC-BY-4.0 License

## Contact

For questions or issues:
- **GitHub Issues:** [Submit an issue](https://github.com/Mahdi-Bashiri/MS3SEG/issues)
- **Email:** [your.email@institution.edu]

## Acknowledgments

This work was supported by [Funding Information]. We thank all patients who participated in this study and provided consent for data sharing.

---

**Last Updated:** [Date]
