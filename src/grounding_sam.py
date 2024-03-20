import argparse
import os
import sys
import rospy
import numpy as np
import json
import torch
from PIL import Image
from sensor_msgs.msg import Image
import time
import cv_bridge
from PIL import Image as PILImage

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anythinglo
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np

class GroundingSam:

    def __init__(self):

        self.realsense_image = None
        # cfg
        self.config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
        self.grounded_checkpoint = "groundingdino_swint_ogc.pth"  # change the path of the model
        self.sam_version = "vit_h"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.sam_hq_checkpoint = None
        self.use_sam_hq = False
        # self.image_path = self.realsense_image##### Make it the most recently received realsense image form the ros topic
        # self.text_prompt = "Traffic cone" #Will eventually be connectec to the LLM output
        self.output_dir = "outputs"
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = "cuda"

        #clear cuda cache
        torch.cuda.empty_cache()

        # rospy.init_node('grounding_sam', anonymous=True)
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.bridge = cv_bridge.CvBridge()
    
    def image_callback(self, data):
        self.realsense_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def load_image(self):
        #convert realsense Image type to a PIL image
        image_pil = PILImage.fromarray(self.realsense_image).convert("RGB")
        
        #save the image to the output directory
        image_pil.save(os.path.join(self.output_dir, 'image.jpg'))

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


    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
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
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases


    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
            print(value + idx + 1)

        # Convert mask_img to a uint8 numpy array before saving
        mask_img_uint8 = mask_img.numpy().astype('uint8')

        # Save mask_img as a PNG
        PILImage.fromarray(mask_img_uint8).save(os.path.join(output_dir, 'mask.png'))
        
        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)


    def main(self, prompt):

        text_prompt = prompt

        # make dir
        os.makedirs(self.output_dir, exist_ok=True)

        # load image
        image_pil, image = self.load_image()
        # load model
        model = self.load_model(self.config_file, self.grounded_checkpoint, device=self.device)

        # visualize raw image
        # name the image to output_dir as raw_image + current time
        raw_image_path = os.path.join(self.output_dir, "raw_image_" + str(time.time()) + ".jpg")

        #start time
        start = time.time()

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            model, image, text_prompt, self.box_threshold, self.text_threshold, device=self.device
        )

        #end time
        end = time.time()
        print(f"Dino time taken: {end - start}")

        # initialize SAM
        if self.use_sam_hq:
            predictor = SamPredictor(sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))
        # image = cv2.imread(self.mage_path)
        #convert realsense image to a numpy array
        image = np.array(self.realsense_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.dtype == np.uint16:
            # Convert from uint16 to uint8 by downscaling by 256
            image = (image / 256).astype(np.uint8)

        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = predictor.predict_torch(
                                            point_coords = None,
                                            point_labels = None,
                                            boxes = transformed_boxes.to(self.device),
                                            multimask_output = False,
                                            )

        #end time
        end = time.time()
        print(f"SAM time taken: {end - start}")

        self.save_mask_data(self.output_dir, masks, boxes_filt, pred_phrases)
        print("Grounding SAM done")
        print('masks.shape:', masks.shape)
        return masks


# if __name__ == '__main__':
#     gs = GroundingSam()
    # while gs.realsense_image is None:
    #     pass
    # gs.main('Traffic cone')
    