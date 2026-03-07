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

# Constants
intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
gt_depth_scale = 256.0
input_size = (616, 1064)
mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
padding = [123.675, 116.28, 103.53]


def load_model():
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_giant2", pretrain=True)
    # model = torch.nn.DataParallel(model)
    model.cuda().eval()
    return model


def fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db):
    model = load_model()
    for image, name in get_next_image_path(path_to_db):
        rgb, pad_info, rgb_origin, scaled_intrinsics = preprocessing(image)
        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({"input": rgb})

        postprocessed_depth = get_depth_image(
            pred_depth, pad_info, rgb_origin, scaled_intrinsics
        )
        postprocessed_normal = get_normal_image(output_dict, pad_info)
        cv2.imwrite(f"{depth_db}/{name}.jpg", postprocessed_depth)
        cv2.imwrite(f"{normal_db}/{name}.jpg", postprocessed_normal)
        print("Done:", name)
        torch.cuda.empty_cache()
        gc.collect()


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
    depth_uint8 = depth_norm.astype(np.uint8)
    return depth_uint8


def get_normal_image(output_dict, pad_info):
    pred_normal = output_dict["prediction_normal"][:, :3, :, :]
    # normal_confidence = output_dict['prediction_normal'][:, 3, :, :]
    pred_normal = pred_normal.squeeze()
    pred_normal = pred_normal[
        :,
        pad_info[0] : pred_normal.shape[1] - pad_info[1],
        pad_info[2] : pred_normal.shape[2] - pad_info[3],
    ]
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    return (pred_normal_vis * 255).astype(np.uint8)


def get_next_image_path(image_db_path):
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    for root, _, files in os.walk(image_db_path):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in extensions:
                file_name = os.path.splitext(file)[0].lower()
                yield os.path.join(root, file), file_name


def preprocessing(rgb_file):
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
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    return rgb, pad_info, rgb_origin, scaled_intrinsic


def cli_command():
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_images> <output_path>")
        sys.exit(1)
    _, path_to_db, target_path = sys.argv
    depth_db = os.path.join(target_path, "depth_db")
    normal_db = os.path.join(target_path, "normal_db")
    os.makedirs(depth_db, exist_ok=True)
    os.makedirs(normal_db, exist_ok=True)
    fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db)


if __name__ == "__main__":
    """
    depth_db = os.path.join(os.path.dirname(__file__), "output/depth")
    normal_db = os.path.join(os.path.dirname(__file__), "output/normal")
    path_to_db = os.path.join(os.path.dirname(__file__), "data/images")

    os.makedirs(depth_db, exist_ok=True)
    os.makedirs(normal_db, exist_ok=True)

    fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db)
  """
    cli_command()
