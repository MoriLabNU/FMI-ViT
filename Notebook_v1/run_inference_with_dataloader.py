# 将本段代码复制到 notebook 中，替换原来的「加载模型」+「推理」两个 cell。
# 使用与命令行 test.sh 相同的 Runner + test_dataloader 流程，避免 inference API 的 list/dtype 问题。

import glob
import shutil
import numpy as np
from PIL import Image
from mmengine.config import Config
from mmengine.runner import Runner

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) 收集要推理的图片路径
if os.path.isfile(INPUT_IMAGE_OR_DIR):
    image_paths = [INPUT_IMAGE_OR_DIR]
else:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(INPUT_IMAGE_OR_DIR, ext)))
    image_paths = sorted(image_paths)

if not image_paths:
    print("未找到图片，请检查 INPUT_IMAGE_OR_DIR 路径。")
else:
    print(f"共 {len(image_paths)} 张图片。")

    # 2) 准备临时数据集目录（与 config 里 test_dataloader 的 data_prefix 一致）
    temp_data_root = os.path.join(project_root, "temp_infer_data")
    img_dir = os.path.join(temp_data_root, "images", "validation")
    ann_dir = os.path.join(temp_data_root, "annotations", "validation")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        dst_img = os.path.join(img_dir, base_name + ext)
        if os.path.abspath(img_path) != os.path.abspath(dst_img):
            shutil.copy2(img_path, dst_img)
        # 占位标注：与图像同名的 .png，全 0（背景），LoadAnnotations 需要存在
        try:
            im = np.array(Image.open(img_path))
            h, w = im.shape[:2]
        except Exception:
            h, w = 512, 512
        seg = np.zeros((h, w), dtype=np.uint8)
        Image.fromarray(seg).save(os.path.join(ann_dir, base_name + ".png"))

    # 3) 加载 config，覆盖 data_root、checkpoint、输出路径
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.launcher = "none"
    cfg.load_from = CHECKPOINT_PATH
    cfg.work_dir = OUTPUT_DIR
    cfg.test_evaluator["output_dir"] = OUTPUT_DIR
    cfg.test_evaluator["keep_results"] = True
    cfg.test_dataloader.dataset.data_root = temp_data_root

    # 4) 打开可视化并指定保存目录
    if "visualization" in cfg.default_hooks:
        cfg.default_hooks["visualization"]["draw"] = True
        cfg.visualizer["save_dir"] = OUTPUT_DIR

    # 5) 构建 Runner 并跑 test（与 tools/test.sh 一致）
    runner = Runner.from_cfg(cfg)
    runner.test()

    print("全部完成。分割结果与可视化已保存到:", OUTPUT_DIR)
