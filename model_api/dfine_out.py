# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from third_party.vlfm.vlfm.vlm.coco_classes import COCO_CLASSES
from third_party.vlfm.vlfm.vlm.detections import ObjectDetections
from model_api.server_wrapper_out import ServerMixin, host_model, send_request_vlm, str_to_image
from PIL import Image
# try:
sys.path.insert(0, "third_party/D-FINE/")
from src.core import YAMLConfig
sys.path.pop(0)
# except Exception:
#     print("Could not import dfine. This is OK if you are only using the client.")

# os.environ["OUT_HOST"] = 

class DFine(nn.Module):
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self._load_model(config_path, checkpoint_path)
        self.model.eval()
        # Warmup
        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, 640, 640).to(self.device)
            with torch.no_grad():
                self.model(dummy_img)

    def _load_model(self, config_path: str, checkpoint_path: str):
        """Loads and configures the D-FINE model."""
        cfg = YAMLConfig(config_path)
        # Handle special cases in config
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
        
        # Build and convert model to deploy mode
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(self.device)
        self.postprocessor = cfg.postprocessor.deploy()

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.4,
        classes: Optional[List[str]] = None,
    ) -> ObjectDetections:
        """
        Performs object detection on the input image.

        Args:
            image: Input image in RGB format as numpy array
            conf_thres: Confidence threshold for detection filtering
            classes: Optional list of classes to filter by (not implemented)
        """
        # Convert numpy array to PIL Image
        im_pil = Image.fromarray(image).convert("RGB")
        orig_size = torch.tensor([image.shape[:2][::-1]]).to(self.device)  # (w, h)

        # Preprocess
        transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        img_tensor = transform(im_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            labels, boxes, scores = self.postprocessor(outputs, orig_size)

        # Filter and convert to numpy
        mask = scores > conf_thres
        labels = labels[mask].cpu().numpy()
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask].cpu().numpy()

        # Convert to normalized coordinates [x1, y1, x2, y2] format
        h, w = image.shape[:2]
        normalized_boxes = boxes.copy()
        normalized_boxes[:, [0, 2]] /= w
        normalized_boxes[:, [1, 3]] /= h

        phrases = [COCO_CLASSES[int(idx)] for idx in labels]
        detections = ObjectDetections(
            boxes=normalized_boxes,
            logits=scores,
            phrases=phrases,
            image_source=image,
            fmt="xyxy",
        )

        return detections


class DFineClient:
    def __init__(self, port: int = 13184):
        self.url = f"http://localhost:{port}/dfine"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request_vlm(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13184) 
    parser.add_argument("--config", type=str, default="third_party/D-FINE/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml")
    parser.add_argument("--checkpoint", type=str, default="pretrained_weights/dfine_x_obj2coco.pth")
    args = parser.parse_args()

    print("Loading model...")

    class DFineServer(ServerMixin, DFine):
        def __init__(self, config: str, checkpoint: str):
            super().__init__(config, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

        def process_payload(self, payload: dict) -> dict:
            # 处理 GET 请求
            if payload is None:  
                return {"status": "ok", "message": "DFine service is running"}
                
            # 处理 POST 请求
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    dfine = DFineServer(args.config, args.checkpoint)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(dfine, name="dfine", port=args.port)
