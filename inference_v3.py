dependencies = ["torch", "torchvision"]
import os
import torch
import cv2
import numpy as np
from memory_profiler import profile
from mmengine import Config, DictAction
import sys
import gc
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from itertools import islice
from rich.progress import track
from monitoring import start_monitor_in_background, SHUTDOWN
import pickle
import tempfile
from concurrent.futures import wait, ALL_COMPLETED
import signal

# Constants
intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
gt_depth_scale = 256.0
input_size = (616, 1064)
mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
padding = [123.675, 116.28, 103.53]

BATCH_SIZE = 16
QUEUE_SIZE = 8
CPU_WORKERS = 4 
WRITE_WORKERS = 2 

log = logging.getLogger()
shutdown = SHUTDOWN

def sigterm_handler(signum, frame):
    shutdown.set()

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def load_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    with torch.no_grad():
        model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True)
        model.cuda(gpu_id).eval()
    return model


def preprocess_single(args):
    rgb_file, name = args
    try:
        rgb_origin = cv2.imread(rgb_file)
        if rgb_origin is None:
            raise ValueError(f"Could not read image: {rgb_file}")
        rgb_origin = rgb_origin[:, :, ::-1]  # BGR -> RGB

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
        rgb_chw = np.ascontiguousarray(rgb.transpose((2, 0, 1)), dtype=np.float32)
        return rgb_chw, pad_info, rgb_origin, scaled_intrinsic, name
    except Exception as e:
        log.error(f"[Producer] Failed to preprocess {name}: {e}")
        return None


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


def get_normal_image(pred_normal_batch, pad_info):
    pred_normal = pred_normal_batch[:, :3, :, :]
    pred_normal = pred_normal.squeeze()  # [3, H, W]
    pred_normal = pred_normal[
        :,
        pad_info[0] : pred_normal.shape[1] - pad_info[1],
        pad_info[2] : pred_normal.shape[2] - pad_info[3],
    ]
    pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
    pred_normal_vis = (pred_normal_vis + 1) / 2
    return (pred_normal_vis * 255).astype(np.uint8)


@profile
def write_worker(write_queue):
    futures = set()
    with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as pool:
        while not shutdown.is_set():
            item = write_queue.get()
            if item is None:
                break
            depth_path, normal_path, depth_img, normal_img = item
            futures.add(pool.submit(cv2.imwrite, depth_path, depth_img))
            futures.add(pool.submit(cv2.imwrite, normal_path, normal_img))
            if len(futures) > 4:
                log.info("queue limit hit")
                _, futures = wait(futures, return_when=ALL_COMPLETED)
    for f in futures:
        f.result()


def _chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def producer(image_list, queues, num_gpus):
    WINDOW = CPU_WORKERS * 4
    batch_buffers = [[] for _ in range(num_gpus)]
    gpu_idx = 0

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as pool:
        for window in _chunked(image_list, WINDOW):
            if shutdown.is_set():
                break
            results = list(pool.map(preprocess_single, window))

            for result in results:
                if shutdown.is_set():
                    break
                if result is None:
                    continue

                gpu_id = gpu_idx % num_gpus
                gpu_idx += 1
                batch_buffers[gpu_id].append(result)

                if len(batch_buffers[gpu_id]) >= BATCH_SIZE:
                    queues[gpu_id].put(list(batch_buffers[gpu_id]))
                    batch_buffers[gpu_id] = []
    for gpu_id in range(num_gpus):
        if batch_buffers[gpu_id]:
            queues[gpu_id].put(list(batch_buffers[gpu_id]))
    for gpu_id in range(num_gpus):
        queues[gpu_id].put(None)


def consumer(gpu_id, queue, depth_db, normal_db, write_queue):
    torch.cuda.set_device(gpu_id)
    log.info(f"[GPU {gpu_id}] Loading model...")
    model = load_model(gpu_id)

    _mean = mean.cuda(gpu_id)
    _std  = std.cuda(gpu_id)

    log.info(f"[GPU {gpu_id}] Model ready.")

    while not shutdown.is_set():
        batch = queue.get()
        if batch is None:
            break

        items = [item for item in batch if item is not None]
        if not items:
            continue

        name = None
        try:
            rgb_nps, pad_infos, rgb_origins, scaled_intrinsics_list, names = zip(*items)
            batch_np = np.stack(rgb_nps, axis=0)
            batch_gpu = (
                torch.from_numpy(batch_np)
                .pin_memory()
                .cuda(gpu_id, non_blocking=True)
                .float()
            )
            batch_gpu = (batch_gpu - _mean) / _std
            pred_depths   = []
            output_dicts  = []
            with torch.no_grad():
                for i in range(len(items)):
                    pred_depth, _, output_dict = model.inference(
                        {"input": batch_gpu[i : i + 1]}
                    )
                    pred_depths.append(pred_depth)
                    output_dicts.append(output_dict)
            for i, (pad_info, rgb_origin, scaled_intrinsics, name) in enumerate(
                zip(pad_infos, rgb_origins, scaled_intrinsics_list, names)
            ):
                depth_img  = get_depth_image(
                    pred_depths[i], pad_info, rgb_origin, scaled_intrinsics
                )
                normal_img = get_normal_image(
                    output_dicts[i]["prediction_normal"],
                    pad_info,
                )
                write_queue.put(
                    (
                        f"{depth_db}/{name}.jpg",
                        f"{normal_db}/{name}.jpg",
                        depth_img,
                        normal_img,
                    ),
                    block=True,
                )
                log.info(f"[GPU {gpu_id}] Done: {name}")

            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"[GPU {gpu_id}] ERROR on {name}: {e}", exc_info=True)

    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"[GPU {gpu_id}] Worker finished.")


def get_next_image_path(image_db_path):
    FILENAME_OF_PATH = "paths.pkl"
    if os.path.exists(FILENAME_OF_PATH):
        log.info(
            f"CACHE HIT: file {FILENAME_OF_PATH} helped save us alot of time. Thanks {FILENAME_OF_PATH}"
        )
        return pickle.load(open(FILENAME_OF_PATH, "rb"))
    list_of_files = []
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    for root, _, files in track(os.walk(image_db_path)):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in extensions:
                file_name = os.path.splitext(file)[0].lower()
                list_of_files.append((os.path.join(root, file), file_name))
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=".") as tmp:
        pickle.dump(list_of_files, tmp)
        tmp_name = tmp.name
        os.replace(tmp_name, FILENAME_OF_PATH)
    return list_of_files


def fetch_normal_and_depth_from_image(path_to_db, depth_db, normal_db):
    start_monitor_in_background()
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")

    log.info(
        f"Found {num_gpus} GPU(s). Batch size: {BATCH_SIZE}, CPU workers: {CPU_WORKERS}"
    )

    done = {os.path.splitext(f)[0] for f in os.listdir(depth_db)}
    all_images = [
        (image, name)
        for image, name in get_next_image_path(path_to_db)
        if name not in done
    ]
    log.info(f"Found {len(all_images)} images to process ({len(done)} already done).")
    if not all_images:
        log.info("Nothing to do.")
        return

    write_queue = Queue(maxsize=QUEUE_SIZE * num_gpus)
    writer = Thread(target=write_worker, args=(write_queue,), daemon=True)
    writer.start()

    queues = [Queue(maxsize=QUEUE_SIZE) for _ in range(num_gpus)]
    consumers = [
        Thread(
            target=consumer,
            args=(gpu_id, queues[gpu_id], depth_db, normal_db, write_queue),
            daemon=True,
        )
        for gpu_id in range(num_gpus)
    ]
    for c in consumers:
        c.start()

    producer(all_images, queues, num_gpus)

    for c in consumers:
        c.join()

    write_queue.put(None)
    writer.join()
    log.info("All done.")


def cli_command():
    if len(sys.argv) < 3:
        print("Usage: python inference_batch.py <path_to_images> <output_path>")
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