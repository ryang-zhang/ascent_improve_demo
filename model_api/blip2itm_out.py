from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from model_api.server_wrapper_out import ServerMixin, host_model, send_request, str_to_image

# try:
from lavis.models import load_model_and_preprocess
# except ModuleNotFoundError:
#     print("Could not import lavis. This is OK if you are only using the client.")

import os
from flask import Flask, request, jsonify

# os.environ["OUT_HOST"] = 

class BLIP2ITM:
    """BLIP 2 Image-Text Matching model."""

    def __init__(
        self,
        name: str = "blip2_image_text_matching",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            cosine = self.model({"image": img, "text_input": txt}, match_head="itc").item()

        return cosine

    def match(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](txt)
        with torch.inference_mode():
            itm_output = self.model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            # print(f'The image and room type {txt} are matched with a probability of {itm_scores[:, 1].item():.3%}')
            match_score = itm_scores[:, 1].item()

        return match_score

class BLIP2ITMClient:
    def __init__(self, port: int = 13182):
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt, method="cosine")
        return float(response["response"])

    def match(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.match: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt, method="match")
        return float(response["response"])
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13182)
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def __init__(self):
            BLIP2ITM.__init__(self)
            ServerMixin.__init__(self)

        def process_payload(self, payload: dict) -> dict:
            if payload is None:  # 处理 GET 请求
                return {"status": "ok", "message": "BLIP2ITM service is running"}
                
            # 处理 POST 请求
            image = str_to_image(payload.get("image"))
            method = payload.get("method", "cosine")
            if method == "cosine":
                response = self.cosine(image, payload["txt"])
            elif method == "match":
                response = self.match(image, payload["txt"])
            else:
                raise ValueError(f"Unsupported method: {method}")
            return {"response": response}

    blip = BLIP2ITMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)