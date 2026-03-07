dependencies = ["torch", "torchvision"]

import os
import torch
import cv2
import numpy as np

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
import time


# Constants
intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
gt_depth_scale = 256.0
input_size = (616, 1064)
mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
padding = [123.675, 116.28, 103.53]

log = logging.getLogger()


def load_model(gpu_id):
    with torch.cuda.device(gpu_id):
        model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2", pretrain=True)
        model.cuda(gpu_id).eval()
    return model


def preprocessing(rgb_file, gpu_id):
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(
        rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )
    scaled_intrinsic = [
        intrinsic[0] * scale,
        intrinsic[1] * scale,
        intrinsic[2] * scale,
        intrinsic[3] * scale,
    ]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    # Move mean/std to the correct GPU
    _mean = mean.cuda(gpu_id)
    _std = std.cuda(gpu_id)
    rgb = torch.div((rgb.cuda(gpu_id) - _mean), _std)
    rgb = rgb[None, :, :, :]
    return rgb, pad_info, rgb_origin, scaled_intrinsic


def get_depth_image(pred_depth, pad_info, rgb_origin, scaled_intrinsics):
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()
    canonical_to_real_scale = scaled_intrinsics[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale
    pred_depth = torch.clamp(pred_depth, 0, 300)
    depth_np = pred_depth.detach().cpu().numpy()
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype(np.uint8)


def get_normal_image(output_dict, pad_info):
    pred_normal = output_dict["prediction_normal"][:, :3, :, :]
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[
        :,
        pad_info[0] : pred_normal.shape[1] - pad_info[1],
        pad_info[2] : pred_normal.shape[2] - pad_info[3],
    ]
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    return (pred_normal_vis * 255).astype(np.uint8)


def process_image(args):
    image_path, name, model, gpu_id, depth_db, normal_db = args
    try:
        with torch.cuda.device(gpu_id):
            rgb, pad_info, rgb_origin, scaled_intrinsics = preprocessing(
                image_path, gpu_id
            )
            with torch.no_grad():
                start = time.perf_counter()
                pred_depth, confidence, output_dict = model.inference({"input": rgb})
            depth_img = get_depth_image(
                pred_depth, pad_info, rgb_origin, scaled_intrinsics
            )
            normal_img = get_normal_image(output_dict, pad_info)
            cv2.imwrite(f"{depth_db}/{name}.jpg", depth_img)
            cv2.imwrite(f"{normal_db}/{name}.jpg", normal_img)
            torch.cuda.empty_cache()
            gc.collect()
            log.info(
                f"[GPU {gpu_id}] Done: {name}. Processing was {time.perf_counter() - start}"
            )
    except Exception as e:
        log.error(f"[GPU {gpu_id}] ERROR on {name}: {e}")


def get_next_image_path(image_db_path):
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    for root, _, files in os.walk(image_db_path):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in extensions:
                file_name = os.path.splitext(file)[0].lower()
                yield os.path.join(root, file), file_name


def fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")

    log.info(f"Loading model on {num_gpus} GPU(s)...")
    models = [load_model(i) for i in range(num_gpus)]
    log.info("Models loaded.")

    all_images = list(get_next_image_path(path_to_db))
    image_in_folder = os.listdir(depth_db)
    all_images = [
        (image, name)
        for image, name in all_images
        if not f"{name}.jpg" in image_in_folder
    ]
    log.info(f"Found {len(all_images)} images to process.")

    # Assign images round-robin to GPUs
    [i for i, (image, name) in enumerate(all_images)]
    total_cores = 0
    for i in range(num_gpus):
        device = torch.cuda.get_device_properties(i)
        cc_cores_per_SM_dict = {
            (2, 0): 32,
            (2, 1): 48,
            (3, 0): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,
            (5, 2): 128,
            (6, 0): 64,
            (6, 1): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 128,
            (8, 9): 128,
            (9, 0): 128,
            (10, 0): 128,
            (12, 0): 128,
        }
        # the above dictionary should result in a value of "None" if a cc match
        # is not found.  The dictionary needs to be extended as new devices become
        # available, and currently does not account for all Jetson devices
        major = device.major
        minor = device.minor
        sm_count = device.multi_processor_count
        cores_per_sm = cc_cores_per_SM_dict.get((major, minor))
        total_cores += cores_per_sm * sm_count

    with ThreadPoolExecutor(max_workers=total_cores) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        for future in as_completed(futures):
            future.result()  # re-raises exceptions if any


def cli_command():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <path_to_images> <output_path>")
        sys.exit(1)
    _, path_to_db, target_path = sys.argv
    depth_db = os.path.join(target_path, "depth_db")
    normal_db = os.path.join(target_path, "normal_db")
    os.makedirs(depth_db, exist_ok=True)
    os.makedirs(normal_db, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(target_path, "inference.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db)


if __name__ == "__main__":
    cli_command()
