# (Quick Start) Segmentation for FMI

This package provides **slide-window inference** for segmenting large biomedical images (e.g. CLSM, histology) using a ViT-S/16 + UPerNet model. No ground truth is required—run inference only and save segmentation visualizations.

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA if you want GPU)
- [mmcv](https://github.com/open-mmlab/mmcv) (>= 2.0.0rc4)
- [mmengine](https://github.com/open-mmlab/mmengine) (>= 0.5.0, < 1.0.0)
- Other runtime deps: `matplotlib`, `numpy`, `Pillow`, etc. (see `requirements.txt`)

Install example:

```bash
pip install torch torchvision
pip install mmcv mmengine
pip install -r requirements.txt
```

---

## Checkpoint
We provide a pretrained segmentation checkpoint for **large biomedical image inference** (slide-window inference, no ground truth required).

### Available Model

| Architecture | Download |
|--------------|----------|
| ViT-S/16 + UPerNet | [iter_8000.pth (full checkpoint)](https://drive.google.com/file/d/1cdzUZwkuQ40LZAWDvYCb9SUwFlzCgnQQ/view?usp=drive_link) |

---

### How to Use

1. Download the checkpoint file.
2. Place it under: `checkpoints/iter_8000.pth`

---

## Quick Start (Notebook)

1. **Open the project folder as your working directory**  
   In Jupyter/Lab or VS Code, open the folder that contains `inference_large_image.ipynb` and set the notebook’s working directory to this folder (e.g. run the first cell so that `project_root` is correct).

2. **Set paths in the notebook (Section 2)**  
   - `CHECKPOINT_PATH`: path to your `.pth` checkpoint (default: `checkpoints/iter_8000.pth`).  
   - `INPUT_IMAGE_OR_DIR`: path to a **single image** or a **folder of images** (e.g. `my_images/`).  
   - `OUTPUT_DIR`: where segmentation visualizations will be saved (e.g. `outputs/`).

3. **Run all cells**  
   - Section 1: add project root to path.  
   - Section 2: print config/checkpoint/input/output (no need to change unless you use different paths).  
   - Section 3: copies your images into a temporary dataset layout, runs the test pipeline (slide inference), and writes results to `OUTPUT_DIR`. No metrics (e.g. IoU) are computed because no ground truth is used.

4. **Check results**  
   Outputs are saved under `OUTPUT_DIR` (e.g. overlay visualizations). No evaluation scores are reported.

---

## Folder Layout

```
.
├── README.md
├── inference_large_image.ipynb   # Main entry for users
├── requirements.txt
├── checkpoints/
│   └── iter_8000.pth             # Your segmentation checkpoint
├── configs/                      # Model and dataset config (no need to edit)
├── mmseg/                        # Model and inference code
├── tools/
├── my_images/                    # Put your images here (or set INPUT_IMAGE_OR_DIR)
└── outputs/                     # Segmentation results (after running the notebook)
```

- **Input:** any image path or folder you set in `INPUT_IMAGE_OR_DIR` (e.g. `my_images/`).  
- **Output:** the path you set in `OUTPUT_DIR` (e.g. `outputs/`). Only visualizations are saved; no metric files.

---

## Supported Image Formats

The notebook looks for files with extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`. Place them in the folder you set as `INPUT_IMAGE_OR_DIR` (or pass a single file path).

---

## Command-Line Alternative

If you prefer the command line and have a dataset laid out as `images/validation/` and `annotations/validation/` (placeholder annotations are fine for inference-only), you can run:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/test.py configs/our/small_upernet_test.py checkpoints/iter_8000.pth \
  --show-dir outputs --work-dir outputs --out outputs
```

For most users, the notebook is simpler because it builds the temporary dataset from `INPUT_IMAGE_OR_DIR` automatically.

---

## Notes

- **No ground truth:** The pipeline is set to **not** compute any evaluation metrics (e.g. IoU). Only inference and visualization are run.
- **Large images:** The model uses **slide inference** (sliding window) so large images are supported without resizing the whole image at once.
- **GPU:** Set `DEVICE` to `"cuda:0"` (or another GPU index) in the notebook if you have a GPU; use `"cpu"` otherwise.

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `No images found` | Check that `INPUT_IMAGE_OR_DIR` points to a folder with images (or a single image file) and that extensions are `.png`, `.jpg`, etc. |
| `Checkpoint not found` | Ensure the checkpoint file exists at `CHECKPOINT_PATH` (e.g. `checkpoints/iter_8000.pth`). |
| `ModuleNotFoundError: mmseg` | Run the notebook from the project root (the folder that contains `mmseg/`) and run the first cell so `project_root` is correct. |
| Out of memory | Use fewer or smaller images per run, or set `DEVICE` to `"cpu"` (slower). |

---

## Citation and Contact

If you use this code or model in your work, please cite the paper/repository as indicated in the main project. For questions or issues, please open an issue in the main repository.
