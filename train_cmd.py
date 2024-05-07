#-*-coding:utf-8-*-

cmds = [
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune10 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.1",
    # "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune20 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.2",
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune30 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.3",
    # "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune40 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.4",
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune50 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.5",
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune60 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.6",
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune70 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.7",
    "python yolov7_train_pruned.py --weights runs/train/ball/weights/best.pt --cfg cfg/training/yolov7-tiny.yaml --hyp data/hyp.scratch.tiny.yaml --batch-size 4 --name ball_prune80 --img-size 960 960 --data datasets/ball_detection/data.yaml --epochs 150 --prun_ratio 0.8",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)