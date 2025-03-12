import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import joblib
import numpy as np
from strategy.rally_checker import RallyChecker
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.pose.general import save_one_box, non_max_suppression as pose_non_max_suppression, scale_coords as pose_scale_coords
from utils.plots import plot_one_box
from utils.pose.plots import colors as pose_colors, plot_skeleton_kpts, plot_one_box as pose_plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from detect_with_ML import Queue

def detect(save_img=False):
    words = "Pending"
    adjacent_frame = 4
    regression_frame = 3
    frame_list = []

    landing_path = r"models\landing_model\landing_0126\AdaBoost_cfg_model.joblib"
    ML_classes = ["flying", "landing"]
    joblib_model = joblib.load(landing_path)

    x_regression_path = r"models\regression_model\Ridge_modelx.joblib"
    x_regressor = joblib.load(x_regression_path)
    y_regression_path = r"models\regression_model\Ridge_modely.joblib"
    y_regressor = joblib.load(y_regression_path)

    pose_weights = opt.pose_weights
    kpt_label = opt.kpt_label
    ball_weights, human_weights = opt.ball_weights, opt.human_weights
    source, view_img, save_txt, imgsz, trace = opt.source, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

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

    rally_checker = RallyChecker()

    # Load pose model
    pose_model = attempt_load(pose_weights, map_location=device)  # load FP32 model
    stride = int(pose_model.stride.max())  # model stride
    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2;
        "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        pose_model.half()  # to FP16

    # Load ball model
    ball_model = attempt_load(ball_weights, map_location=device)  # load FP32 model
    stride = int(ball_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        ball_model = TracedModel(ball_model, device, opt.img_size)
    if half:
        ball_model.half()  # to FP16

    # Load ball model
    human_model = attempt_load(human_weights, map_location=device)  # load FP32 model
    stride = int(human_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        human_model = TracedModel(human_model, device, opt.img_size)
    if half:
        human_model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    pose_names = pose_model.module.names if hasattr(pose_model, 'module') else pose_model.names
    human_names = human_model.module.names if hasattr(human_model, 'module') else human_model.names
    ball_names = ["ball"]
    names = [pose_names, ball_names, human_names]
    # red, green, blue, gray
    human_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (169, 169, 169)]
    # Purple
    ball_color = [(128, 0, 128)]
    colors = [pose_colors, ball_color, human_colors]
    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    BoxProcessor = Queue(max_length=adjacent_frame * 2 + 1, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    BoxRegProcessor = Queue(max_length=regression_frame, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    play_duration = 3
    # play_threshold = 0.8

    #这一行是初始化的结束
    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        humans_box, humans_action = [], []
        ball_exist = False
        ball_center = (-1, -1)
        poses = []
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
                ball_model(img, augment=opt.augment)[0]
                human_model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pose_pred = pose_model(img, augment=opt.augment)[0]
            ball_pred = ball_model(img, augment=opt.augment)[0]
            human_pred = human_model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pose_pred = pose_non_max_suppression(pose_pred, opt.pose_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        ball_pred = non_max_suppression(ball_pred, opt.ball_thres, opt.iou_thres, classes=opt.classes,agnostic=opt.agnostic_nms)
        human_pred = non_max_suppression(human_pred, opt.human_thres, opt.iou_thres, classes=opt.classes,agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        preds = [pose_pred, ball_pred, human_pred]
        types = ["poses", "ball", "humans"]
        # colors = [ball_color, human_colors]
        for pred, color, name, typ in zip(preds, colors, names, types):
            # if typ == "poses":
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    if typ == 'poses':
                        # Rescale boxes from img_size to im0 size
                        pose_scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                        pose_scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {name[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for det_index, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or opt.save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = f'{name[c]} {conf:.2f}'
                                kpts = det[det_index, 6:]  # 51/3 =17个点
                                height = 720
                                right_elbow = kpts[24:26].tolsit()
                                right_wrist = kpts[30:32].tolist()
                                left_elbow = kpts[21:23].tolist()
                                left_wrist = kpts[27:29].tolist()
                                poses = [right_elbow, right_wrist, left_elbow, left_wrist]
                                x1, y1 = right_elbow[0], right_elbow[1]
                                x2, y2 = right_wrist[0], right_wrist[1]
                                if y1 > height / 2 and y1 > y2:
                                    print(f'right_elbow:{right_elbow}, right_wrist:{right_wrist}, elbow_y:{y1}, wrist_y:{y2}')
                                    cv2.putText(im0, 'lower_hit', (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                pose_plot_one_box(xyxy, im0, label=label, color=color(c, True), line_thickness=1, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                    else:
                        # Rescale boxes from img_size to im0 size
                        scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {name[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in det[:, :6]:
                            if typ == "ball":
                                ball_exist = True
                                ball_center = ([(xyxy[0].tolist() + xyxy[2].tolist()) / 2,
                                                (xyxy[1].tolist() + xyxy[3].tolist()) / 2])
                            else:
                                humans_box.append([i.tolist() for i in xyxy])
                                humans_action.append(name[int(cls)])
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{name[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=color[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        BoxProcessor.enqueue(ball_pred[0])
        BoxRegProcessor.enqueue(ball_pred[0])

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
            pred_ball_location = [int(ball_next_real_x[0]), int(ball_next_real_y[0])]
            cv2.circle(im0, (int(ball_next_real_x[0]), int(ball_next_real_y[0])), 5, (0, 255, 0), -1)
        else:
            pred_ball_location = (-1,-1)

        rally_checker.process(ball_exist, ball_center, pred_ball_location, humans_box, humans_action, frame, words, poses)
        frame_list.append(im0)
        rally_checker.visualize(im0) #把检测的 h,w 传进去， 绘图

        if idx >= adjacent_frame:
            if not BoxProcessor.check_enough():
                words = "Pending"
            else:
                ball_locations = BoxProcessor.get_queue()
                words = ML_classes[
                    int(joblib_model.predict(np.expand_dims(np.array(ball_locations).flatten(), axis=0))[0])]

            color = (0, 0, 255) if words == "landing" else (0, 255, 0)
            cv2.putText(im0, words, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # display_img = frame_list[0]
            # cv2.imshow(str(p), display_img)
            # frame_list = frame_list[1:]

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

    rally_checker.output_csv(opt.output_csv_folder, opt.output_csv_file)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_weights', nargs='+', type=str, default="models\pose_model\last.pt", help='model.pt path(s)')
    parser.add_argument('--ball_weights', nargs='+', type=str, default=r"models\ball_model\ball_0210\best.pt", help='model.pt path(s)')
    parser.add_argument('--human_weights', nargs='+', type=str, default=r"models\human_model\human_1211\best.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"source\2s\2s.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument("--output_csv_folder", default=r"landing_csv\2s")
    parser.add_argument("--output_csv_file", default="2s.csv")
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--pose-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--ball-thres', type=float, default=0.5, help='ball confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--human-thres', type=float, default=0.25, help='human confidence threshold')
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
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
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
