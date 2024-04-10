import argparse
import os

import numpy as np
import json
import torch
import torchvision
from PIL import Image as PILImage
import litellm

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
) 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Recognize Anything Model & Tag2Text
from recognize_anything.ram.models import ram
from recognize_anything.ram import inference_ram
import torchvision.transforms as TS
import sensor_msgs.msg
import rospy
import cv_bridge

# ChatGPT or nltk is required when using tags_chineses
# import openai
# import nltk

class RAM:

    def __init__(self):

        # cfg
        self.config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py' # change the path of the model config file
        self.ram_checkpoint = 'pretrained/ram_swin_large_14m.pth'  # change the path of the model
        self.grounded_checkpoint = "pretrained/groundingdino_swint_ogc.pth"  # change the path of the model
        self.sam_checkpoint = "pretrained/sam_vit_h_4b8939.pth"
        # self.sam_hq_checkpoint = args.sam_hq_checkpoint
        self.use_sam_hq = None
        # self.image_path = args.input_image
        # self.split = args.split
        # self.openai_proxy = args.openai_proxy
        self.output_dir = "outputs"
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.device = "cuda"

        #clear cuda cache
        torch.cuda.empty_cache()

        # rospy.init_node('grounding_sam', anonymous=True)
        self.sub = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.msg.Image, self.image_callback)
        self.bridge = cv_bridge.CvBridge()

        #load model
        self.model = self.load_model(self.config_file, self.grounded_checkpoint, device=self.device)

    def image_callback(self, data):
        self.realsense_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def load_image(self):
        # load image
        image_pil = PILImage.fromarray(self.realsense_image).convert("RGB")

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold,device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases


    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)


    def save_mask_data(self, output_dir, tags_chinese, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = {
            'tags_chinese': tags_chinese,
            'mask':[{
                'value': value,
                'label': 'background'
            }]
        }
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data['mask'].append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'label.json'), 'w') as f:
            json.dump(json_data, f)
        

    def main(self):

        # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        # load image
        image_pil, image = self.load_image(self.image_path)
        # load model
        model = self.model

        # visualize raw image
        image_pil.save(os.path.join(self.output_dir, "raw_ram_image.jpg"))

        # initialize Recognize Anything Model
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
                        TS.Resize((384, 384)),
                        TS.ToTensor(), normalize
                    ])
        
        # load model
        ram_model = ram(pretrained=self.ram_checkpoint,
                                            image_size=384,
                                            vit='swin_l')
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        ram_model.eval()

        ram_model = ram_model.to(self.device)
        raw_image = image_pil.resize(
                        (384, 384))
        raw_image  = transform(raw_image).unsqueeze(0).to(self.device)

        res = inference_ram(raw_image , ram_model)

        # Currently ", " is better for detecting single tags
        # while ". " is a little worse in some case
        tags=res[0].replace(' |', ',')

        print("Image Tags: ", res[0])

        # run grounding dino model
        boxes_filt, scores, pred_phrases = self.get_grounding_output(
            model, image, tags, self.box_threshold, self.text_threshold, device=self.device
        )

        # initialize SAM
        if self.use_sam_hq:
            print("Initialize SAM-HQ Predictor")
            predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        self.save_mask_data(self.output_dir, tags_chinese, masks, boxes_filt, pred_phrases)
