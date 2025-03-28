import argparse
import time
import os
from scripts.json2csv import json_to_csv
import json
from strategy.json_analysis import main as json_analysis
from strategy.data_manager import DataManagement
from strategy.tracker.sort_tracker import BoxTracker
from pathlib import Path
from utils.click_court import click_points
from strategy.classifier.image_classifier import ImageClassifier
import cv2
import torch
import torch.backends.cudnn as cudnn
from strategy.court.court_detector import CourtDetector
import joblib
import numpy as np
from collections import deque
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.pose.general import save_one_box, non_max_suppression as pose_non_max_suppression, scale_coords as pose_scale_coords
from utils.plots import plot_one_box
from utils.pose.plots import colors as pose_colors, plot_skeleton_kpts, plot_one_box as pose_plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from strategy.court.top_view import TopViewProcessor
import shutil

top_view_h, top_view_w = 1000, 480

class Queue:
    def __init__(self, max_length, h, w, no_exist_with_prev=False):
        self.max_length = max_length
        self.queue = deque()
        self.h, self.w = h, w
        self.no_exist_with_prev = no_exist_with_prev

    def process_item(self, item):
        if len(item) == 0:
            self.queue.append(self.queue[-1])

    def enqueue(self, item):
        if self.no_exist_with_prev:
            if -1 in item:
                if self.queue:
                    self.queue.append(self.queue[-1])
                else:
                    self.queue.append((-1, -1))
            else:
                self.queue.append((item[0] / self.w, item[1] / self.h))
        else:
            self.queue.append((item[0] / self.w, item[1] / self.h))
        # if len(item) == 0:
        #     if self.queue:
        #         self.queue.append(self.queue[-1])
        #     else:
        #         self.queue.append((-1, -1))
        # else:
        #     self.queue.append((item[0] / self.w, item[1] / self.h))

        if len(self.queue) > self.max_length:
            self.queue.popleft()

    def dequeue(self):
        return self.queue.popleft()

    def get_length(self):
        return len(self.queue)

    def is_empty(self):
        return len(self.queue) == 0

    def get_queue(self):
        return list(self.queue)

    def check_enough(self):
        return self.get_length() == self.max_length
    


def detect():
    adjacent_frame = 4
    regression_frame = 3
    frame_list, tv_list = [], []

    mask_points_str = opt.masks
    if mask_points_str:
        mask_pre = mask_points_str[0].split(' ')
        mask_points = [(int(mask_pre[0]), int(mask_pre[1])), (int(mask_pre[2]), int(mask_pre[3])),
                       (int(mask_pre[4]), int(mask_pre[5])), (int(mask_pre[6]), int(mask_pre[7]))]
    else:
        mask_points = []


    classifier_path = "weights/latest_assets/mobilenet/best_acc.pth"
    model_cfg = "/".join(classifier_path.split("/")[:-1]) + "/model_cfg.yaml"
    label_path = "/".join(classifier_path.split("/")[:-1]) + "/labels.txt"
    highlight_classifier = ImageClassifier(classifier_path, model_cfg, label_path, device="cuda:0")

    landing_path = 'weights/latest_assets/landing/latest_landing.joblib'
    curve_class_file = os.path.join(os.path.dirname(landing_path), "classes.txt")
    with open(curve_class_file, 'r') as file:
        curve_class = [line[:-1] for line in file.readlines()]
    curve_model = joblib.load(landing_path)
    ball_curve_color = [(0, 255, 0), (0, 0, 255)]

    x_regression_path = 'weights/latest_assets/x_regression.joblib'
    x_regressor = joblib.load(x_regression_path)
    y_regression_path = 'weights/latest_assets/y_regression.joblib'
    y_regressor = joblib.load(y_regression_path)

    seqML_classifier = "weights/latest_assets/seqML/model.joblib"
    seqML_model = joblib.load(seqML_classifier)
    seqML_class = "weights/latest_assets/seqML/classes.txt"
    seqML_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0,0,0)]
    with open(seqML_class, 'r') as file:
        seq_ML_classes = [line[:-1] for line in file.readlines()]
    seq_ML_config = os.path.join(os.path.dirname(seqML_classifier), "config.json")
    with open(seq_ML_config, 'r') as file:
        seq_ML_config = json.load(file)
        seq_num = seq_ML_config["seq_num"]
        seq_step = seq_ML_config["seq_step"]

    pose_weights = opt.pose_weights
    ball_weights = opt.ball_weights
    source, view_img, save_txt, imgsz, trace, use_saved_box,player_num = (
        opt.source, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.use_saved_box, opt.player_num)
    tracker = BoxTracker()

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

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

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    pose_names = pose_model.module.names if hasattr(pose_model, 'module') else pose_model.names
    ball_names = ["ball"]
    names = [ball_names, pose_names]
    click_type = 'detect' # 'detect' or 'inner'
    keep_court = False

    ball_color = [(128, 0, 128)]
    # frame_id = 0
    colors = [ball_color, pose_colors]

    data_manger = DataManagement(player_num=player_num)
    top_view = TopViewProcessor(players=player_num)
            # DataManagement.first_filter(box_assets=box_assets['{}'.format(frame_id)])

    t0 = time.time()

    BoxProcessor = Queue(max_length=adjacent_frame * 2 + 1, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    BoxRegProcessor = Queue(max_length=regression_frame, h=dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            w=dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH), no_exist_with_prev=True)

    cap = cv2.VideoCapture(source)
    ret, img = cap.read()
    img_h, img_w = img.shape[:2]
    directory_path = os.path.dirname(source)

    output_folder = opt.output_folder
    if output_folder:
        os.makedirs(opt.output_folder, exist_ok=True)
        topview_video = os.path.join(opt.output_folder, "topview.mp4")
        output_video = os.path.join(opt.output_folder, "output.mp4")
        topview_writer = cv2.VideoWriter(topview_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (top_view_w, top_view_h))
        output_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_w, img_h))


    if not mask_points:
        mask_points = click_points(img)
    cap.release()
    cv2.destroyAllWindows()

    box_asset_path = os.path.join(directory_path, os.path.basename(source).split(".")[0] + ".json")
    box_assets_filter_path = os.path.join(output_folder, os.path.basename(source).split(".")[0] + "_filter.json")
    if use_saved_box:
        box_asset_path = os.path.join(directory_path, os.path.basename(source).split(".")[0] + ".json")
        assert os.path.exists(box_asset_path), "The box asset file does not exist."
        with open(box_asset_path, 'r') as f:
            box_assets = json.load(f)
        mask_points = box_assets['mask']
        click_type = box_assets['mask_type']
    else:
        box_assets = {}
        # if os.path.exists(box_asset_path):
        #     input("The box asset file already exists, do you want to overwrite it? Press Enter to continue, or Ctrl+C to exit.")
        box_f = open(box_asset_path, 'w')
        box_assets['mask'] = mask_points
        box_assets['mask_type'] = click_type

    box_f_filter = open(box_assets_filter_path,'w')
    court_detector = CourtDetector(mask_points)
    init_lines = court_detector.begin(type=click_type, frame=img, mask_points=mask_points)
    init_matrix = court_detector.game_warp_matrix[-1]
    classifier_list =[]
    classifier_status_list = []
    # central_y, central_x = int((init_lines[9] + init_lines[11])//2), int((init_lines[-12] + init_lines[-10])//2)

    for idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if idx < opt.start_with:
            continue
        if idx == opt.stop_at:
            break

        if use_saved_box:
            classifier_result = int(box_assets[str(idx)]["classifier"])
            classifier_status = classifier_result
        else:
            box_assets[idx] = {}
            classifier_result = highlight_classifier(im0s).tolist() # 0: playing, 1: highlight
            classifier_list.append(classifier_result)
            box_assets[idx]["classifier"] = classifier_result

            play_duration = 5  # 0: play 1:high
            if len(classifier_list) < play_duration:
                play_num = classifier_list.count(0)
                if play_num < len(classifier_list) * 0.8:
                    classifier_status = 1
                else:
                    classifier_status = 0
            else:
                play_actions_duration = classifier_list[-play_duration:]
                play_num = play_actions_duration.count(0)
                if play_num < play_duration * 0.8:
                    classifier_status = 1
                else:
                    classifier_status = 0
            classifier_status_list.append(classifier_status)
            # box_assets[idx] = {}
            # classifier_result = highlight_classifier(im0s).tolist() # 0: playing, 1: highlight
            # box_assets[idx]["classifier"] = classifier_result
        if classifier_status == 0:
            ball_center = []
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            if use_saved_box:
                t1 = time_synchronized()
                ball_pred = [torch.tensor(box_assets[str(idx)]["ball"])]
                pose_pred = [torch.tensor(box_assets[str(idx)]["person"])]
                t2 = time_synchronized()
                t3 = time_synchronized()

            else:
                t1 = time_synchronized()
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    ball_pred = ball_model(img, augment=opt.augment)[0]
                    pose_pred = pose_model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # Apply NMS
                ball_pred = non_max_suppression(ball_pred, opt.ball_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                pose_pred = pose_non_max_suppression(pose_pred, opt.pose_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=True)
                box_assets[idx]["ball"] = ball_pred[0].tolist() #.cpu().tolist()
                box_assets[idx]["person"] = pose_pred[0].tolist()#.cpu().tolist()
                t3 = time_synchronized()

            preds = [ball_pred, pose_pred] # detection result
            types = ["ball", "pose"]

            for pred, color, name, typ in zip(preds, colors, names, types):
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if len(det):
                        if typ == 'pose':
                            # pose_output = det.cpu().tolist()
                            pose_scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                            pose_scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)

                            tracked_pose = tracker.update(det.cpu())
                            # Print results
                            for c in det[:, 5].unique():
                                n = (det[:, 5] == c).sum()  # detections per class
                                s += f"{n} {name[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for det_index, (*xyxy, id, cls) in enumerate(tracked_pose[:, :6]):
                                # humans_box.append([i.tolist() for i in xyxy])
                                # humans_action.append(name[int(cls)])

                                c = int(cls)  # integer class
                                label = f'id: {id:.2f}, {name[c]} '
                                kpts = det[det_index, 6:]  # 51/3 =17个点
                                pose_plot_one_box(xyxy, im0, label=label, color=color(c, True), line_thickness=1,
                                                  kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        else:
                            scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {name[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in det[:, :6]:
                                if len(xyxy):
                                    ball_center.append([(xyxy[0].tolist() + xyxy[2].tolist()) / 2,
                                                        (xyxy[1].tolist() + xyxy[3].tolist()) / 2])
                                else:
                                    ball_center.append([-1,-1])
                                label = f'{name[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=color[int(cls)], line_thickness=1)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Predefined lines for classifier_status[-1] == 0 cases
            if len(classifier_status_list) == 1:
                court_detector = CourtDetector(mask_points)
                lines = court_detector.detect(frame=im0s, mask_points=mask_points)
            else:
                try:
                    if classifier_status_list[-2] == 0:
                        lines = court_detector.track_court(frame=im0s, mask_points=mask_points)
                        # court_detector = CourtDetector(mask_points)
                        # lines = court_detector.detect(frame=im0s, mask_points=mask_points)
                    elif classifier_status_list[-2] == 1:
                        court_detector = CourtDetector(mask_points)
                        lines = court_detector.detect(frame=im0s, mask_points=mask_points)
                except:
                    lines = init_lines
            if lines is None:
                lines = init_lines
                # current_matrix
            try:
                current_matrix = court_detector.game_warp_matrix[-1]
            except:
                current_matrix = init_matrix
            highlight_classifier.visualize(im0, classifier_result)

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

            data_assets = {}
            data_assets["person"] = tracked_pose
            data_assets["ball"] = ball_center
            data_assets["classifier"] = classifier_result
            data_assets["court"] = lines
            data_assets["ball_prediction"] = pred_ball_location

            # data_manger.first_filter(data_assets)
            data_manger.first_filter(data_assets,current_matrix,idx)
            box_assets_filter = data_manger.real_players

            real_ball = data_manger.real_balls[-1]
            real_players = data_manger.real_players[-1]
            top_view_img = top_view.visualize_bv(real_ball, real_players)
            court_detector.visualize(im0, lines)

            ball_location = data_manger.get_ball()
            BoxProcessor.enqueue(ball_location)
            BoxRegProcessor.enqueue(ball_location)

            if not BoxProcessor.check_enough():
                curve_status = "Pending"
                if -1 not in ball_location:
                    cv2.putText(im0, curve_status, (int(ball_location[0]), int(ball_location[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                ball_locations = BoxProcessor.get_queue()
                curve_cls = int(curve_model.predict(np.expand_dims(np.array(ball_locations).flatten(), axis=0))[0])
                curve_status = curve_class[curve_cls]
                if -1 not in ball_location:
                    cv2.putText(im0, curve_status, (int(ball_location[0]), int(ball_location[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, ball_curve_color[curve_cls], 2)

            kps = data_manger.get_kps_prediction_input(seq_step, seq_num)
            if kps:
                seqML_result = seqML_model.predict(np.array(kps)).tolist()
                for _, (result, player_loc) in enumerate(zip(seqML_result, data_manger.get_player_foot_pixels())):
                    i = int(result)
                    cv2.putText(im0, seq_ML_classes[i], (int(player_loc[0]), int(player_loc[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, seqML_colors[i], 2)
            else:
                seqML_result = [-1 for _ in range(player_num)]

            data_manger.second_update(landing=curve_status, kps_pred=seqML_result) # pending -1, flying 0,landing 1
            strategy_assets = data_manger.get_strategy_assets()

            # Visualize
            tv_list.append(cv2.resize(top_view_img, (top_view_w, top_view_h)))

        else:
            box_assets[idx] = {}
            classifier_result = highlight_classifier(im0s).tolist()  # 0: playing, 1: highlight
            highlight_classifier.visualize(im0s, classifier_result)

            classifier_list.append(classifier_result)
            box_assets[idx]["classifier"] = classifier_result
            data_assets = {}
            data_assets["person"] = -1
            data_assets["ball"] = -1
            data_assets["classifier"] = -1
            data_assets["court"] = -1
            data_assets["ball_prediction"] = -1
            strategy_assets = data_manger.get_strategy_assets_dummy(idx)

        frame_list.append(im0s)
        # tv_list.append()
        tv_list.append(cv2.resize(top_view.visualize_dummy(), (top_view_w, top_view_h)))

        if idx >= adjacent_frame:
            display_img = frame_list[0]
            topview_img = tv_list[0]
            if not opt.no_show:
                cv2.imshow("Top View", topview_img)
                cv2.imshow(str(p), display_img)

            if output_folder:
                output_writer.write(display_img)
                topview_writer.write(topview_img)

            del frame_list[0]
            del tv_list[0]

            if not opt.no_show:
                cv2.waitKey(opt.wait_key)

    if not use_saved_box:
        json.dump(box_assets, box_f, indent=4)
        box_f.close()
        json.dump(strategy_assets, box_f_filter, indent=4)
        box_f_filter.close()

    csv_file_path = os.path.join(output_folder, os.path.basename(source).split(".")[0] + ".csv")
    json_to_csv(box_assets_filter_path, csv_file_path)
    shutil.copy(source, os.path.join(output_folder, os.path.basename(source)))
    shutil.copy(box_asset_path, os.path.join(output_folder, os.path.basename(box_asset_path)))
    json_analysis(box_assets_filter_path, source, output_folder, "info.json")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_weights', nargs='+', type=str, default="weights/latest_assets/yolopose_4lr.pt", help='model.pt path(s)')
    parser.add_argument('--ball_weights', nargs='+', type=str, default="weights/latest_assets/ball.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r"C:\Users\User\Desktop\game1.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument("--output_folder", default="output_whole/game1")
    # parser.add_argument("--output_csv_file", default="2s.csv")
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
    parser.add_argument('--wait_key', type=int, default=1, help='wait key for cv2.waitKey')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--masks',default='',nargs='+', help='mask')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint labels')
    parser.add_argument("--use_saved_box",action="store_true",help="Load box json for fast inference")
    parser.add_argument('--stop_at', type=int, default=-1, help='')
    parser.add_argument('--start_with', type=int, default=-1, help='')
    parser.add_argument('--player_num', type=int, default=2, help='')
    parser.add_argument('--no_show', action='store_true', help='update all models')


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
