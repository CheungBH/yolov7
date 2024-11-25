from .CNN_models import ModelBuilder
from .utils import read_labels, image_normalize, crop, scale
import torch
from .kps_vis import KeyPointVisualizer
import yaml
import cv2


class ImageClassifier:
    def __init__(self, weight, config, label, transform=None, device="cuda:0", max_batch=4):
        self.transform = transform
        self.label = label
        self.parse_config(config)
        self.classes = read_labels(label)
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), self.backbone, device)
        self.MB.load_weight(weight)
        self.model.eval()
        self.max_batch = max_batch
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        if transform is not None:
            self.KPV = KeyPointVisualizer(transform.kps, "coco")

    def parse_config(self, config_file):
        with open(config_file, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.backbone = config["model"]["backbone"]
        self.model_size = config["model"]["input_size"]

        try:
            self.img_type = config["image_type"]
            assert self.img_type in ["black_crop", "raw_crop", 'black_whole', 'raw_whole'], \
                "Unsupported image type: {}".format(self.img_type)
        except:
            self.img_type = "raw_whole"

    def __call__(self, img, boxes=None, kps=None, kps_exist=None):
        img_tns = self.preprocess(img, boxes, kps, kps_exist)
        scores = self.MB.inference_tensor(img_tns)
        self.scores = scores[0]
        _, self.pred_idx = torch.max(self.scores, 0)
        self.pred_cls = self.classes[self.pred_idx]
        self.color = self.colors[self.pred_idx]
        self.vis_str = "{}: {}".format(self.pred_cls, self.scores[self.pred_idx].item())
        return self.pred_cls

    def preprocess(self, img, boxes, kps, kps_score):
        if "crop" in self.img_type:
            img = img if self.img_type == "raw_crop" else self.KPV.visualize(img, kps, kps_score)
            imgs_tensor = None
            for box in boxes:
                scaled_box = self.transform.scale(img, box)
                cropped_img = self.transform.SAMPLE.crop(scaled_box, img)
                img_tensor = image_normalize(cropped_img, size=self.model_size)
                imgs_tensor = torch.unsqueeze(img_tensor, dim=0) if imgs_tensor is None else torch.cat(
                    (imgs_tensor, torch.unsqueeze(img_tensor, dim=0)), dim=0)
        elif "whole" in self.img_type:
            if "raw" in self.img_type:
                target_img = img
            else:
                target_img = self.KPV.visualize(img, kps, kps_score)
            img_tensor = image_normalize(target_img, size=self.model_size)
            imgs_tensor = torch.unsqueeze(img_tensor, dim=0)
        else:
            raise ValueError("Unknown image type: {}".format(self.img_type))
        return imgs_tensor

    def visualize(self, frame):
        w = frame.shape[1]
        cv2.putText(frame, self.vis_str, (w-1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)




