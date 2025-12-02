#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import Tuple
import os

import rospkg

# Class mapping: red=0, yellow=1, green=2, greenleft=3, redleft=4
IDX2LABEL = ["red", "yellow", "green", "greenleft", "redleft"]
LABEL2IDX = {n: i for i, n in enumerate(IDX2LABEL)}

prep = transforms.Compose([
    transforms.Resize((20, 70)),  # (H, W)
    transforms.ToTensor(),        # [0,1]
])

class NIA_SEGNet_module(nn.Module):
     def __init__(self):
         super().__init__()
         self.fcn = models.mobilenet_v2(pretrained=True)
         self.fcn.classifier = nn.Sequential(
             nn.Dropout(0.2),
             nn.Linear(self.fcn.last_channel, 5),
         )
     def forward(self, x): return self.fcn(x)

def load_classifier(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model = NIA_SEGNet_module()
    new_state = {}
    for k, v in state.items():
        if k.startswith("fcn.") or k.startswith("fc."):
            new_state[k] = v
        elif k.startswith("model."):
            new_state[k.replace("model.", "")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval().to(device)
    return model

def xyxy_to_int_bbox(xyxy: torch.Tensor, w: int, h: int, pad: int = 2) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    x1 = max(0, int(np.floor(x1) - pad))
    y1 = max(0, int(np.floor(y1) - pad))
    x2 = min(w - 1, int(np.ceil(x2) + pad))
    y2 = min(h - 1, int(np.ceil(y2) + pad))
    return x1, y1, x2, y2

def draw_label(img_bgr: np.ndarray, text: str, x1: int, y1: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (x1 + 2, y1 - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def color_for_label(name: str):
    if name.startswith("green"): return (0, 255, 0)
    if name.startswith("yellow"): return (0, 255, 255)
    return (0, 0, 255)

class TrafficLightRosNode:
    def __init__(self):
        rospy.init_node('traffic_light_inference_node', anonymous=True)
        rp = rospkg.RosPack()
        PKG = 'traffic_pkg'
        pkg_path = rp.get_path(PKG)

        # Params
        self.image_topic      = rospy.get_param("~image_topic", "/camera_3_undistorted/compressed")
        self.yolo_weights     = rospy.get_param("~yolo_weights", "best.pt")
        self.classifier_ckpt  = rospy.get_param("~classifier_ckpt", "mobilenet.ckpt")
        self.traffic_class_id = int(rospy.get_param("~traffic_class_id", 1))
        self.det_conf_thres   = float(rospy.get_param("~det_conf_thres", 0.25))
        self.crop_pad         = int(rospy.get_param("~crop_pad", 2))
        self.pub_det_topic    = rospy.get_param("~pub_detections_topic", "/traffic/detections")
        self.pub_img_topic    = rospy.get_param("~pub_image_topic", "/traffic/image_bbox/compressed")

        self.yolo_weights = os.path.join(pkg_path, 'models', self.yolo_weights)
        self.classifier_ckpt = os.path.join(pkg_path, 'models', self.classifier_ckpt)
        
        # Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"[Traffic] Device: {self.device}")
        self.yolo = YOLO(self.yolo_weights)
        self.cls_model = load_classifier(self.classifier_ckpt, self.device)

        # ROS I/O
        self.pub_dets = rospy.Publisher(self.pub_det_topic, Float32MultiArray, queue_size=1)
        self.pub_img  = rospy.Publisher(self.pub_img_topic, CompressedImage, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("[Traffic] Ready. Subscribed to %s", self.image_topic)
        rospy.spin()

    @torch.no_grad()
    def image_callback(self, msg: CompressedImage):
        # Decode
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            rospy.logwarn("[Traffic] Failed to decode image")
            return
        h, w = img_bgr.shape[:2]
        vis = img_bgr.copy()

        # YOLO
        res = self.yolo(img_bgr)[0]
        out = []  # Float32MultiArray payload

        if res.boxes is not None and len(res.boxes) > 0:
            xyxys = res.boxes.xyxy
            confs = res.boxes.conf
            clses = res.boxes.cls

            for i in range(len(xyxys)):
                yolo_cls = int(clses[i].item())
                if yolo_cls != self.traffic_class_id:
                    continue
                box_conf = float(confs[i].item())
                if box_conf < self.det_conf_thres:
                    continue

                x1, y1, x2, y2 = xyxy_to_int_bbox(xyxys[i], w, h, pad=self.crop_pad)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Classifier
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(crop_rgb)
                t = prep(pil).unsqueeze(0).to(self.device)
                logits = self.cls_model(t)
                prob = F.softmax(logits, dim=1)[0]
                cls_id = int(torch.argmax(prob).item())    # state id
                cls_conf = float(prob[cls_id].item())      # state prob
                cls_name = IDX2LABEL[cls_id]

                # Draw
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_for_label(cls_name), 2)
                draw_label(vis, f"{cls_name} ({cls_conf:.2f})", x1, y1)

                # [class_id, box_conf, cls_conf, x1, y1, x2, y2]
                out.extend([
                    float(cls_id), float(box_conf), float(cls_conf),
                    float(x1), float(y1), float(x2), float(y2)
                ])

        # Publish detections
        arr = Float32MultiArray()
        arr.data = out
        self.pub_dets.publish(arr)

        # Publish overlay image
        vis_msg = CompressedImage()
        vis_msg.header.stamp = rospy.Time.now()
        vis_msg.format = "jpeg"
        vis_msg.data = np.array(cv2.imencode(".jpg", vis)[1]).tobytes()
        self.pub_img.publish(vis_msg)

if __name__ == "__main__":
    try:
        TrafficLightRosNode()
    except rospy.ROSInterruptException:
        pass
