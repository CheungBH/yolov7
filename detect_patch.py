import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import torch

def remove_boundary_boxes(bboxes, image_size, distance=20):
    """
    Remove bounding boxes that are close to the image boundaries.

    Parameters:
    - bboxes (torch.Tensor): A tensor of shape (N, 6) where each row contains
                             [x1, y1, x2, y2, conf, cls].
    - image_size (tuple): A tuple (width, height) representing the image size.
    - distance (int): The minimum distance from the boundary to keep the box.

    Returns:
    - torch.Tensor: A tensor of bounding boxes that are not close to the boundary.
    """
    width, height = image_size
    keep_mask = (
        (bboxes[:, 0] >= distance) &  # x1
        (bboxes[:, 1] >= distance) &  # y1
        (bboxes[:, 2] <= width - distance) &  # x2
        (bboxes[:, 3] <= height - distance)  # y2
    )
    return bboxes[keep_mask]


def post_nms(boxes, iou_threshold=0.8):
    # return boxes

    if boxes.numel() == 0:
        return torch.empty((0, 6), device=boxes.device)  # Return empty tensor if no boxes

    # Sort boxes by confidence score in descending order
    boxes = boxes[torch.argsort(boxes[:, 4], descending=True)]

    selected_boxes = []
    while boxes.size(0) > 0:
        chosen_box = boxes[0]  # Select the highest confidence box
        selected_boxes.append(chosen_box)

        if boxes.size(0) == 1:
            break  # No more boxes left to compare

        other_boxes = boxes[1:]

        # Compute IoU (Intersection over Union)
        x1 = torch.maximum(chosen_box[0], other_boxes[:, 0])
        y1 = torch.maximum(chosen_box[1], other_boxes[:, 1])
        x2 = torch.minimum(chosen_box[2], other_boxes[:, 2])
        y2 = torch.minimum(chosen_box[3], other_boxes[:, 3])

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        chosen_area = (chosen_box[2] - chosen_box[0]) * (chosen_box[3] - chosen_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        iou = intersection / (chosen_area + other_areas - intersection)

        # Keep boxes below the IoU threshold OR different class
        boxes = other_boxes[(iou < iou_threshold) | (chosen_box[5] != other_boxes[:, 5])]

    return torch.stack(selected_boxes)

def patch_crop(image, target_height, cropped_width=640):
    crop_size = (target_height, target_height)
    img_height, img_width = image.shape[:2]
    target_width = int(target_height * img_width / img_height)
    cropped_images = []

    target_img = cv2.resize(image, (target_width, target_height))
    top_left = [0, target_width - cropped_width]
    for idx in top_left:
        cropped_img = target_img[:, idx:idx + cropped_width]
        cropped_images.append(cropped_img)
    return cropped_images, top_left

def to_original_coord(preds, tls):
    merged_preds = []
    for pred, tl in zip(preds, tls):
        pred[:, 0] += tl
        pred[:, 2] += tl
        merged_preds.append(pred.detach().cpu())
    return merged_preds


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

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
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        patches, tls = patch_crop(im0s, 720, 852)
        batch_imgs = None

        for patch in patches:
            patch = letterbox(patch, new_shape=imgsz, stride=stride)[0]  # resize
            patch = patch[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            patch = np.ascontiguousarray(patch)
            patch = torch.from_numpy(patch).to(device)
            patch = patch.half() if half else patch.float()  # uint8 to fp16/32
            patch /= 255.0  # 0 - 255 to 0.0 - 1.0
            patch = patch.unsqueeze(0)
            batch_imgs = patch if batch_imgs is None else torch.cat((batch_imgs, patch), dim=0)

        img = batch_imgs
        if half:
            img = img.half()

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        final_preds = torch.zeros(1, 6)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        scaled_coords = []
        for p, p_img, tl, im in zip(pred, patches, tls, img):
            p[:, :4] = scale_coords(im.shape[1:], p[:, :4], p_img.shape).round()
            remove_boundary_boxes(p, (852, 720))
            p[:, 0] += tl
            p[:, 2] += tl

            scaled_coords.append(p.detach().cpu())
        for output in scaled_coords:
            final_preds = torch.cat((final_preds, output), dim=0)
        pred = [post_nms(final_preds[1:], 0.8)]
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
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
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
