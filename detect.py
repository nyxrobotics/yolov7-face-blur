#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Object Crop Using YOLOv7
import argparse
import subprocess
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.datasets import LoadStreams
from utils.general import apply_classifier
from utils.general import check_img_size
from utils.general import check_imshow
from utils.general import increment_path
from utils.general import non_max_suppression
from utils.general import scale_coords
from utils.general import set_logging
from utils.general import strip_optimizer
from utils.general import xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier
from utils.torch_utils import select_device
from utils.torch_utils import time_synchronized

WEIGHT_URLS = {
    "yolov7-lite-t.pt": "https://drive.google.com/uc?export=download&id=1HNXd9EdS-BJ4dk7t1xJDFfr1JIHjd5yb",
    "yolov7-lite-s.pt": "https://drive.google.com/uc?export=download&id=1MIC5vD4zqRLF_uEZHzjW_f-G3TsfaOAf",
    "yolov7-tiny.pt": "https://drive.google.com/uc?export=download&id=1Mona-I4PclJr5mjX1qb8dgDeMpYyBcwM",
    "yolov7s-face.pt": "https://drive.google.com/uc?export=download&id=1_ZjnNF_JKHVlq41EgEqMoGE2TtQ3SYmZ",
    "yolov7-face.pt": "https://drive.google.com/uc?export=download&id=1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo",
    "yolov7-face-tta.pt": "https://drive.google.com/uc?export=download&id=1oIaGXFd4goyBvB1mYDK24GLof53H9ZYo",
    "yolov7-w6-face.pt": "https://drive.google.com/uc?export=download&id=1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS",
    "yolov7-w6-face-tta.pt": "https://drive.google.com/uc?export=download&id=1U_kH7Xa_9-2RK2hnyvsyMLKdYB0h4MJS"
}


def download_weights(weight_name, save_dir="weights"):
    """Download weights using gdown for Google Drive files."""
    if weight_name not in WEIGHT_URLS:
        raise ValueError(
            f"Weight '{weight_name}' not found in predefined list.")

    url = WEIGHT_URLS[weight_name]
    save_path = Path(save_dir) / weight_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"Weight file '{save_path}' already exists.")
        return save_path

    print(f"Downloading {weight_name} from Google Drive using gdown...")

    try:
        subprocess.run(
            [
                "gdown", "--id", url.split("id=")[-1], "-O", str(save_path)
            ],
            check=True
        )
        print(f"Downloaded {weight_name} to {save_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading {weight_name}: {e}")
        raise RuntimeError(
            f"Failed to download {weight_name} from Google Drive.")

    return save_path


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, blurratio, hidedetarea, weights_dir = \
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.blurratio, opt.hidedetarea, opt.weights_dir

    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower(
    ).startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = False

    # Load model
    weights_path = Path(weights_dir) / weights[0]
    model = attempt_load(
        str(weights_path), map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Extract only the first four values for coordinates
                    if len(xyxy) < 4:
                        print(
                            f"Unexpected bounding box format: {xyxy}. Skipping.")
                        continue

                    # Use the first four values for bounding box
                    x1, y1, x2, y2 = map(int, xyxy[:4])

                    # Replace negative values with 0
                    x1 = min(max(0, x1), im0.shape[1])
                    y1 = min(max(0, y1), im0.shape[0])
                    x2 = min(max(x1, x2), im0.shape[1])
                    y2 = min(max(y1, y2), im0.shape[0])

                    crop_obj = im0[y1:y2, x1:x2]

                    try:
                        # Apply blurring to the cropped object
                        blur = cv2.blur(crop_obj, (blurratio, blurratio))
                        im0[y1:y2, x1:x2] = blur
                    except Exception as e:
                        print(f"Error while blurring object: {e}")
                        continue

                    # ..................................................................

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (
                            cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if not hidedetarea:
                            plot_one_box(xyxy, im0, label=label,
                                         color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov7-w6-face-tta.pt', help='model.pt path(s)')
    parser.add_argument('--weights-dir', type=str, default='weights',
                        help='directory containing model weights')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    parser.add_argument('--blurratio', type=int, default=20,
                        required=True, help='blur opacity')
    parser.add_argument('--hidedetarea', action='store_true',
                        help='Hide Detected Area')
    opt = parser.parse_args()
    print(opt)

    # Check and download weights if necessary
    for weight in opt.weights:
        weight_path = Path(opt.weights_dir) / weight
        if not weight_path.exists():
            print(
                f"Model file '{weight_path}' not found. Attempting to download...")
            download_weights(weight, save_dir=opt.weights_dir)

    # Proceed with detection or model update
    with torch.no_grad():
        if opt.update:  # Update specified models to fix SourceChangeWarning
            for weight in opt.weights:
                weight_path = Path(opt.weights_dir) / weight
                print(f"Updating model: {weight_path}")
                detect()
                strip_optimizer(str(weight_path))
        else:
            detect()
