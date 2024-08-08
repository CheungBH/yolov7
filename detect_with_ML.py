import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import joblib
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


adjacent_frame = 3
regression_frame = 3

from collections import deque


class Queue:
    def __init__(self, max_length, h, w):
        self.max_length = max_length
        self.queue = deque()
        self.h, self.w = h, w

    def process_item(self, item):
        if len(item) == 0:
            self.queue.append(self.queue[-1])

    def enqueue(self, item):
        if len(item) == 0:
            self.queue.append(self.queue[-1])
        elif len(item) == 1:
            self.queue.append(self.xyxy2center(item[0, :4].tolist()))
        else:
            self.queue.append(self.xyxy2center(item[0, :4].tolist()))

        if len(self.queue) > self.max_length:
            self.queue.popleft()

    def dequeue(self):
        return self.queue.popleft()

    def get_length(self):
        return len(self.queue)

    def xyxy2center(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2)/self.w, ((y1 + y2) / 2)/self.h

    def is_empty(self):
        return len(self.queue) == 0

    def get_queue(self):
        return list(self.queue)

    def check_enough(self):
        return self.get_length() == self.max_length


def detect(save_img=False):
    frame_list = []
    landing_path = "/home/sailhku/Downloads/hard_1.3/landing_inference/GBDT_cfg_model.joblib"
    ML_classes = ["flying", "landing"]
    joblib_model = joblib.load(landing_path)

    x_regression_path = "/home/sailhku/Downloads/RandomForestRegressor_modelx.joblib"
    x_regressor = joblib.load(x_regression_path)
    y_regression_path = "/home/sailhku/Downloads/RandomForestRegressor_modely.joblib"
    y_regressor = joblib.load(y_regression_path)

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

    BoxProcessor = Queue(max_length=adjacent_frame * 2 + 1, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    BoxRegProcessor = Queue(max_length=regression_frame, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                             w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # BoxRegProcessorY = Queue(max_length=regression_frame, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    #                          w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
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
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

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
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

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

            # if view_img:
            frame_list.append(im0)

            BoxProcessor.enqueue(det)
            BoxRegProcessor.enqueue(det)
            # BoxRegProcessor.enqueue(det)

            if BoxRegProcessor.check_enough():
                ball_locations = BoxRegProcessor.get_queue()
                ball_x = np.array([b[0] for b in ball_locations])
                ball_x = np.expand_dims(ball_x, axis=0)
                ball_y = np.array([b[1] for b in ball_locations])
                ball_y = np.expand_dims(ball_y, axis=0)
                ball_next_x = x_regressor.predict(ball_x)
                ball_next_y = y_regressor.predict(ball_y)
                ball_next_real_x = ball_next_x * dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                ball_next_real_y = ball_next_y * dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cv2.circle(im0, (int(ball_next_real_x[0]), int(ball_next_real_y[0])), 10, (0, 255, 0), -1)
                # cv2.putText(im0, f"Next position: ({ball_next_x[0]:.2f}, {ball_next_y[0]:.2f})", (100, 100),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if idx >= adjacent_frame:
                if not BoxProcessor.check_enough():
                    words = "Pending"
                else:
                    ball_locations = BoxProcessor.get_queue()
                    words = ML_classes[int(joblib_model.predict(np.expand_dims(np.array(ball_locations).flatten(), axis=0))[0])]

                color = (0, 255, 0) if words == "landing" else (0, 0, 255)
                cv2.putText(im0, words, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                cv2.imshow(str(p), frame_list[0])
                cv2.waitKey(0)  # 1 millisecond
                frame_list = frame_list[1:]
                # Save results (image with detections)

                if save_img:
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
