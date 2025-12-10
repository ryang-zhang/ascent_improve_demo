import numpy as np
import torch
import torchvision.transforms.functional as F
import os
from model_api.server_wrapper_out import ServerMixin, host_model, send_request_vlm, str_to_image
from PIL import Image
import argparse
import numpy as np
import sys

# try:
sys.path.insert(0, "third_party/recognize-anything/")
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
sys.path.pop(0)
# except ModuleNotFoundError:
#     print("Could not import ram. This is OK if you are only using the client.")

# os.environ["OUT_HOST"] =

RAM_CHECKPOINT_PATH = "pretrained_weights/ram_plus_swin_large_14m.pth"


class RAM:
    def __init__(
        self,
        config_path: str = RAM_CHECKPOINT_PATH,
        image_size: str = 384,
        device: torch.device = torch.device("cuda"),
    ):
        '''
        * The Recognize Anything Plus Model (RAM++)
        '''
        self.transform = get_transform(image_size=image_size)

        #######load model
        self.model = ram_plus(pretrained=config_path,
                                image_size=384,
                                vit='swin_l')
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

        # image = transform(Image.open(args.image)).unsqueeze(0).to(device)

        # res = inference(image, model)

    def predict(self, image: np.ndarray) -> str:
        
        image_source = Image.fromarray(image.astype('uint8')).convert("RGB")
        image_transformed = self.transform(image_source).unsqueeze(0).to(self.device)
        
        result = inference(image_transformed, self.model)

        return result


class RAMClient:
    def __init__(self, port: int = 13185):
        host = os.getenv("OUT_HOST", "localhost")
        self.url = f"http://{host}:{port}/ram"

    def predict(self, image_numpy: np.ndarray) -> str:
        response = send_request_vlm(self.url, image=image_numpy)
        return response[0]


if __name__ == "__main__":

    # True Use
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13185)
    args = parser.parse_args()

    print("Loading model...")

    class RAMServer(ServerMixin, RAM):
        def process_payload(self, payload: dict) -> dict:
            # 处理 GET 请求
            if payload is None:  
                return {"status": "ok", "message": "Ram service is running"}
                
            # 处理 POST 请求

            image = str_to_image(payload["image"])
            return self.predict(image)

    ram = RAMServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(ram, name="ram", port=args.port)
