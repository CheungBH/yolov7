import argparse
import time
from pathlib import Path
from strategy.manager import StrategyManager
from strategy.classifier.image_classifier import ImageClassifier
import cv2
import torch
import torch.backends.cudnn as cudnn
import joblib
from strategy.court.top_view import TopViewProcessor
from strategy.court.court_detector import CourtDetector
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from detect_with_ML import Queue

serve_side, serve_position = "lower", "right"
begin_with = "serve"
ball_locations_list=[]

if serve_side == 'lower':
    ball_init_toward, ball_last_hit = "up", "lower"
else:
    ball_init_toward, ball_last_hit = "down", "upper"


def detect(save_img=False):
    words = "Pending"
    adjacent_frame = 3
    regression_frame = 3
    frame_list = []
    top_view_frame_list = []
    speed_list = []
    heatmap_list = []
    check_stage_list=[]

    classifier_path = "/media/hkuit164/Backup/yolov7/datasets/ball_combine/highlight/highlight.pth"
    model_cfg = "/".join(classifier_path.split("/")[:-1]) + "/model_cfg.yaml"
    label_path = "/".join(classifier_path.split("/")[:-1]) + "/labels.txt"
    highlight_classifier = ImageClassifier(classifier_path, model_cfg, label_path, device="cuda:0") #"cuda:0"


    landing_path = "datasets/ball_combine/landing_model/Ada.joblib"
    ML_classes = ["flying", "landing"]
    joblib_model = joblib.load(landing_path)

    x_regression_path = "datasets/ball_combine/regression_model/Ridge_modelx.joblib"
    x_regressor = joblib.load(x_regression_path)
    y_regression_path = "datasets/ball_combine/regression_model/Ridge_modely.joblib"
    y_regressor = joblib.load(y_regression_path)

    source, view_img, save_txt, imgsz, trace = opt.source, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    ball_weights, human_weights = opt.ball_weights, opt.human_weights
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
    ball_model = attempt_load(ball_weights, map_location=device)  # load FP32 model
    stride = int(ball_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        ball_model = TracedModel(ball_model, device, opt.img_size)
    if half:
        ball_model.half()  # to FP16

    # Load model
    human_model = attempt_load(human_weights, map_location=device)  # load FP32 model
    stride = int(human_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        human_model = TracedModel(human_model, device, opt.img_size)
    if half:
        human_model.half()  # to FP16


    #click_type = "inner/detect"
    mask_points_str = opt.masks  # "374,133 949,143 1152,584 124,582"
    if mask_points_str:
        mask_pre = mask_points_str[0].split(' ')
        mask_points = [(int(mask_pre[0]), int(mask_pre[1])), (int(mask_pre[2]), int(mask_pre[3])),
                       (int(mask_pre[4]), int(mask_pre[5])), (int(mask_pre[6]), int(mask_pre[7]))]
    else:
        mask_points = []
    click_type = 'detect'
    keep_court = False

    def click_points():
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mask_points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('image', img)
                if len(mask_points) > 4:
                    mask_points.pop(0)
                    print(mask_points)

        height, width, channel = img.shape
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", width, height)
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_event)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        print(mask_points)

    #mask_points = []
    cap = cv2.VideoCapture(source)
    ret, img = cap.read()
    # h, w = img.shape[:2]
    # img = cv2.resize(img, (h/2, w/2))

    if not mask_points:
        click_points()
    cap.release()

    court_detector = CourtDetector(mask_points)
    init_lines = court_detector.begin(type=click_type, frame=img, mask_points=mask_points)
    central_y, central_x = int((init_lines[9] + init_lines[11])//2), int((init_lines[-12] + init_lines[-10])//2)
    # rally_checker = RallyChecker(central_x=int(central_x), central_y=int(central_y))

    strategies = StrategyManager(check_stage=begin_with, serve_side=serve_side,
                                 serve_position=serve_position, ball_last_hit=ball_last_hit,
                                 central_x=central_x, central_y=central_y, ball_init_toward=ball_init_toward, )
    top_view = TopViewProcessor(2)

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    human_names = human_model.module.names if hasattr(human_model, 'module') else human_model.names
    ball_names = ["ball"]
    names = [ball_names, human_names]
    # red, green, blue, gray
    human_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (169, 169, 169)]
    # Purple
    ball_color = [(128, 0, 128)]
    colors = [ball_color, human_colors]
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
    cls_hp=[]
    play_duration = 3
    #play_threshold = 0.8
    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        classifier_result = highlight_classifier(im0s)
        cls_hp.append(classifier_result)
        recent_actions = cls_hp[-(play_duration+1):]
        play_actions = [True if action == "play" else False for action in recent_actions]
        r1 = play_actions[:play_duration]
        r2 = play_actions[1:]
        if len(cls_hp) < play_duration+1:
            lines = court_detector.track_court(im0s, keep_court=keep_court)
            classifier_status = 'play'
        else:
            if sum(r1) >= play_duration and sum(r2) >=play_duration:
                lines = court_detector.track_court(im0s, keep_court=keep_court)
                classifier_status = 'play'
            elif sum(r2) <play_duration:
                lines = court_detector.track_court(im0s, keep_court=True)
                classifier_status = 'highlight'
            elif sum(r1) < play_duration and sum(r2) >=play_duration:
                court_detector = CourtDetector(mask_points)
                classifier_status = 'play'
                lines = court_detector.detect(frame=im0s, mask_points=mask_points)

        # lines = court_detector.begin(type=click_type, frame=im0s, mask_points=mask_points) if idx == 0 else \
        #     court_detector.track_court(im0s, keep_court=keep_court)

        humans_box, humans_action = [], []
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
            ball_pred = ball_model(img, augment=opt.augment)[0]
            human_pred = human_model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        ball_pred = non_max_suppression(ball_pred, opt.ball_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        human_pred = non_max_suppression(human_pred, opt.human_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        preds = [ball_pred, human_pred]
        types = ["ball", "humans"]

        # colors = [ball_color, human_colors]
        for pred, color, name, typ in zip(preds, colors, names, types):
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
                        s += f"{n} {name[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if typ == "ball":
                            ball_exist = True
                            ball_center = ([(xyxy[0].tolist() + xyxy[2].tolist()) / 2,
                                            (xyxy[1].tolist() + xyxy[3].tolist()) / 2])
                        else:
                            humans_box.append([i.tolist() for i in xyxy])
                            humans_action.append(name[int(cls)])
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{name[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=color[int(cls)], line_thickness=1)
                else:
                    ball_exist = False
                    ball_center = (-1, -1)
            # Print time (inference + NMS)
            inference_time = (t2 - t1) * 1000
            nms_time = (t3 - t2) * 1000
            elapsed_time = inference_time + nms_time
            print(f'{s}Done. ({elapsed_time:.3f}ms)')
        human_realbox = top_view.get_player_location()
        ball_realbox = top_view.get_ball_location()
        csv_path = 'test_csv/gameset1.csv'
        strategies.process(ball_exist, ball_center, humans_box, humans_action,classifier_status, lines,frame, words, human_realbox, ball_realbox,csv_path)
        # strategies.update_line(lines)

        highlight_classifier.visualize(im0)
        # top_view.visualize(im0)
            # Stream results
        frame_list.append(im0)
        if classifier_status == "play":
            strategies.visualize_strategies(im0)
            court_detector.visualize(im0, lines)
            player_bv, _, speed, heatmap = top_view.process(court_detector, strategies.get_box(), strategies.get_ball(),elapsed_time, vis_graph=False)
            top_view_frame_list.append(cv2.resize(player_bv, (480, 640)))
            speed_list.append(speed)
            heatmap_list.append(heatmap)
        else:
            top_view_frame_list.append(top_view_frame_list[-1])
            speed_list.append(speed_list[-1])
            heatmap_list.append(heatmap_list[-1])

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
            cv2.circle(im0, (int(ball_next_real_x[0]), int(ball_next_real_y[0])), 10, (0, 255, 0), -1)

        if idx >= adjacent_frame:
            if not BoxProcessor.check_enough():
                words = "Pending"
            else:
                ball_locations = BoxProcessor.get_queue()
                words = ML_classes[
                    int(joblib_model.predict(np.expand_dims(np.array(ball_locations).flatten(), axis=0))[0])]

            if words == 'landing' and recent_actions[-1] == 'play':
                if ball_locations[-4][1] > 0.5:
                    ball_locations_list.append(1)
                else:
                    ball_locations_list.append(0)
                if len(ball_locations_list) >= 2:
                    if ball_locations_list[-1] == ball_locations_list [-2]:
                        cv2.putText(im0, words, (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            color = (0, 0, 255) if words == "landing" else (0, 255, 0)
            cv2.putText(im0, words, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            display_img = frame_list[0]
            cv2.imshow(str(p), display_img)
            frame_list = frame_list[1:]

            cv2.imshow("Top View", top_view_frame_list[0])
            top_view_frame_list = top_view_frame_list[1:]

            cv2.imshow("Speed", speed_list[0])
            speed_list = speed_list[1:]

            cv2.imshow("Heatmap", heatmap_list[0])
            heatmap_list = heatmap_list[1:]

            cv2.waitKey(1)  # 1 millisecond

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
                vid_writer.write(display_img)

        # if view_img:
        #     cv2.imshow(str(p), im0)
        #     cv2.waitKey(0)  # 1 millisecond
        #
        #
        # # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'image':
        #         cv2.imwrite(save_path, im0)
        #         print(f" The image with the result is saved in: {save_path}")
        #     else:  # 'video' or 'stream'
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer
        #             if vid_cap:  # video
        #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             else:  # stream
        #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
        #                 save_path += '.mp4'
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #         vid_writer.write(im0)



    print(f'Done. ({time.time() - t0:.3f}s)')
    #strategies.rally_checker.output_csv()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ball_weights', nargs='+', type=str, default='models\models1211/ball/best.pt', help='model.pt path(s)')
    parser.add_argument('--human_weights', nargs='+', type=str, default='models\models1211/human/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='video/01.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
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
    parser.add_argument('--masks',default='',nargs='+', help='mask') #"310 162 938 171 1147 601 140 591"
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
