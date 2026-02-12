# MS3SEG Repository Setup Guide

This guide explains how to set up and use the MS3SEG repository for your manuscript.

## 📋 Complete Repository Structure

Your repository is now organized as follows:

```
MS3SEG/
├── preprocessing/          # ✅ Data preprocessing scripts (you mentioned finalized)
│   ├── anonymization.py
│   ├── dicom_to_nifti.py
│   ├── standardization.py
│   └── coregistration_and_brain_extraction.py
│
├── models/                 # ✅ COMPLETE - Model architectures
│   ├── __init__.py        # Model registry and factory function
│   ├── unet.py            # U-Net implementation
│   ├── unet_plusplus.py   # U-Net++ with nested skip connections
│   ├── unetr.py           # UNETR with transformer encoder
│   └── swin_unetr.py      # Swin UNETR with shifted windows
│
├── training/              # ✅ COMPLETE - Training orchestration
│   ├── train.py           # Main training script with k-fold CV
│   ├── evaluate.py        # Comprehensive evaluation script
│   └── config.json        # Configuration template
│
├── utils/                 # ✅ COMPLETE - Helper functions
│   ├── __init__.py
│   ├── data_loader.py     # MS3SEGDataLoader class
│   ├── metrics.py         # Dice, IoU, HD95 + loss functions
│   └── visualization.py   # MS3SEGVisualizer class
│
├── splits/                # Cross-validation splits
│   └── 5fold_splits.json  # (to be created from your data)
│
├── figures/               # Generated figures will go here
├── tables/                # Generated tables will go here
│
├── requirements.txt       # ✅ Python dependencies
├── LICENSE               # ✅ MIT License
├── README.md             # ✅ Comprehensive documentation
├── .gitignore            # ✅ Git ignore rules
└── SETUP_GUIDE.md        # ✅ This file
```

## 🚀 Quick Start Steps

### 1. Initial Setup

```bash
# Clone your repository (or initialize if starting fresh)
git clone https://github.com/Mahdi-Bashiri/MS3SEG.git
cd MS3SEG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your preprocessed data in the following structure:

```
data/
├── patient_001/
│   ├── 001_FLAIR_preprocessed.nii.gz
│   ├── 001_T1_preprocessed.nii.gz
│   ├── 001_T2_preprocessed.nii.gz
│   └── 001_mask.nii.gz
├── patient_002/
│   └── ...
...
├── patient_100/
    └── ...
```

### 3. Update Configuration

Edit `training/config.json`:

```json
{
  "data": {
    "data_dir": "path/to/your/data",  // ← UPDATE THIS
    "target_size": [256, 256]
  },
  ...
}
```

### 4. Create Cross-Validation Splits

```python
from utils import MS3SEGDataLoader

data_loader = MS3SEGDataLoader(data_dir="path/to/data")
all_patients = [f"{i:03d}" for i in range(1, 101)]

splits = data_loader.create_kfold_splits(
    all_patients,
    n_splits=5,
    test_size=0.2,
    random_state=42,
    save_path="splits/5fold_splits.json"
)
```

### 5. Train Models

```bash
# Train all models on multi-class segmentation
python training/train.py --scenario multi_class

# Train specific model
python training/train.py --scenario binary_abnormal_wmh --models U-Net

# Custom config
python training/train.py --config custom_config.json --scenario three_class
```

### 6. Evaluate Models

```bash
# Evaluate all models
python training/evaluate.py --results_dir results/experiment_YYYYMMDD_HHMMSS

# Evaluate specific scenario with visualizations
python training/evaluate.py \
    --results_dir results/experiment_YYYYMMDD_HHMMSS \
    --scenario multi_class \
    --visualize
```

## 🔧 How This Differs From Your Original Code

### ✨ Key Improvements

1. **Modular Structure**
   - Your original: Everything in one 2000+ line file
   - Now: Organized into logical modules (models/, utils/, training/)

2. **Reusable Components**
   - Your original: Code tied to specific workflow
   - Now: Each model, metric, and utility is standalone and importable

3. **Easier Testing**
   - Your original: Hard to test individual components
   - Now: Each module can be tested independently

4. **Better Collaboration**
   - Your original: Merge conflicts with single large file
   - Now: Multiple team members can work on different modules

5. **Publication Ready**
   - Professional structure matches top research repositories
   - Clear documentation and examples
   - Easy for reviewers to verify your methods

## 📊 Mapping Your Original Code

Here's where components from your `models_runner.py` ended up:

| Original Location | New Location | Notes |
|------------------|--------------|-------|
| Model definitions | `models/*.py` | Each architecture in separate file |
| Loss functions | `utils/metrics.py` | Organized with evaluation metrics |
| Data loading | `utils/data_loader.py` | MS3SEGDataLoader class |
| Visualization | `utils/visualization.py` | MS3SEGVisualizer class |
| Training loop | `training/train.py` | MS3SEGTrainer class |
| Evaluation | `training/evaluate.py` | MS3SEGEvaluator class |
| Config class | `training/config.json` | JSON format for flexibility |

## 🎯 Using Your Existing Models

If you have already trained models with your original code:

### Option 1: Retrain with New Code (Recommended)
- Ensures consistency with publication
- Clean, reproducible results
- Takes advantage of improved structure

### Option 2: Load Existing Models
```python
from tensorflow import keras
from utils import MS3SEGEvaluator

# Load your old model
model = keras.models.load_model('path/to/old_model.h5', compile=False)

# Use new evaluation code
evaluator = MS3SEGEvaluator(results_dir='results/existing')
# ... evaluate ...
```

## 🧪 Testing Individual Components

### Test a Model
```python
from models import build_unet

model = build_unet(input_shape=(256, 256, 1), num_classes=4)
model.summary()
```

### Test Data Loading
```python
from utils import MS3SEGDataLoader

loader = MS3SEGDataLoader(data_dir="path/to/data")
images, masks = loader.load_dataset(['001', '002'], scenario='multi_class')
print(f"Loaded {len(images)} slices")
```

### Test Metrics
```python
from utils import calculate_multiclass_metrics
import numpy as np

# Dummy data
y_true = np.random.randint(0, 4, (10, 256, 256))
y_pred = np.random.randint(0, 4, (10, 256, 256))

metrics = calculate_multiclass_metrics(y_true, y_pred, num_classes=4)
print(metrics)
```

## 📝 Creating Figures for Your Paper

```python
from utils import MS3SEGVisualizer
import numpy as np

viz = MS3SEGVisualizer(save_dir='figures')

# Your FLAIR image and mask
image = ...  # Load your actual data
mask = ...

# Generate publication-quality figure
viz.plot_sample_with_mask(
    image, 
    mask, 
    title="Representative MS Patient with Tri-Mask Annotation",
    save_name="figure2_sample_annotation"
)
```

## 🔄 Next Steps for Publication

1. **Test the Pipeline**
   ```bash
   # Small test run
   python training/train.py --scenario binary_ventricles --models U-Net
   ```

2. **Generate All Results**
   ```bash
   # Full experiments for paper
   for scenario in multi_class three_class binary_abnormal_wmh binary_ventricles; do
       python training/train.py --scenario $scenario
   done
   ```

3. **Create Figures**
   ```bash
   python training/evaluate.py --results_dir results/latest --visualize
   ```

4. **Update Paper**
   - Reference code in GitHub repo
   - Include DOI for dataset
   - Cite software dependencies

## ❓ FAQ

**Q: Can I use my existing preprocessed data?**  
A: Yes! Just make sure it follows the expected structure in `data/`.

**Q: Do I need to retrain all models?**  
A: Not necessary, but recommended for consistency with the published code.

**Q: How do I add my own model?**  
A: Create `models/my_model.py`, implement `build_my_model()`, register in `__init__.py`.

**Q: Can I modify the loss function?**  
A: Yes! Edit `training/config.json` or add new loss in `utils/metrics.py`.

**Q: How do I handle class imbalance?**  
A: The code automatically calculates class weights. Adjust in `config.json` if needed.

## 📧 Getting Help

If you encounter issues:
1. Check this guide
2. Review example usage in each module's `__main__` section
3. Check GitHub issues
4. Contact: m_bashiri99@sut.ac.ir

## ✅ Checklist for Repository Release

- [ ] Add your preprocessing scripts to `preprocessing/`
- [ ] Create `splits/5fold_splits.json` from your data
- [ ] Test training pipeline with one model
- [ ] Generate sample figures
- [ ] Update `README.md` with your results
- [ ] Add any additional documentation
- [ ] Create GitHub release
- [ ] Link in manuscript

Good luck with your publication! 🎉
