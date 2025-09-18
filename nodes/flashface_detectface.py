import copy
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageDraw
from pathlib import Path

from ..flashface.all_finetune.utils import PadToSquare, get_padding
from ..ldm.models.retinaface import crop_face, RetinaFace
from ..ldm.utils import load_model_weights
from ..flashface.models.model_registry import get_model_path as get_registry_model_path

padding_to_square = PadToSquare(224)
retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

# Create the retinaface model
retinaface_model = RetinaFace(backbone='resnet50').to('cuda')

# Load pretrained weights using the model registry
model_path = get_registry_model_path('retinaface', auto_download=True)
if model_path is None:
    # Fallback to the legacy path
    model_path = str(Path(__file__).parents[2] / "models" / "facedetection" / "retinaface_resnet50.pth")
    print(f"Warning: Using fallback RetinaFace model path: {model_path}")

print(f"Loading RetinaFace model from: {model_path}")
model_weights = load_model_weights(model_path, device='cuda')
retinaface_model.load_state_dict(model_weights, strict=True)
retinaface_model.eval().requires_grad_(False)

class FlashFaceDetectFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("PIL_IMAGE", "IMAGE")  # Updated RETURN_TYPES
    INPUT_IS_LIST = True
    FUNCTION = "detect_face"
    CATEGORY = "FlashFace"

    def detect_face(self, images):
        pil_imgs = []
        tensor_imgs = []

        # if images are batched, separate them into individual images
        if len(images) == 1 and len(images[0].shape) == 4:
            images = images[0]

        for img in images:
            if not isinstance(img, torch.Tensor):
                raise ValueError("Input should be a list of PIL images")
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            pil_image = F.to_pil_image(img)
            pil_imgs.append(pil_image)
            tensor_imgs.append(img)

        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # detection
        imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to('cuda')
        boxes, kpts = retinaface_model.detect(imgs, min_thr=0.6)

        face_imgs = []
        tensor_face_imgs = []

        for i in range(b):
            scale = 640 / max(pil_imgs[i].size)
            left, top, _, _ = get_padding(round(scale * pil_imgs[i].width),
                                          round(scale * pil_imgs[i].height), 640)

            boxes[i][:, [0, 2]] -= left
            boxes[i][:, [1, 3]] -= top
            kpts[i][:, :, 0] -= left
            kpts[i][:, :, 1] -= top

            boxes[i][:, :4] /= scale
            kpts[i][:, :, :2] /= scale

            crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
            if len(crops) != 1:
                raise ValueError(
                    f'Found {len(crops)} faces in the image {i + 1}, please ensure there is only one face in each image'
                )

            face_imgs += crops
            tensor_face_imgs += [tensor_imgs[i]] * len(crops)

            draw = ImageDraw.Draw(vis_pil_imgs[i])
            for box in boxes[i]:
                box = box[:4].tolist()
                box = [int(x) for x in box]
                draw.rectangle(box, outline='red', width=4)

        # Permute tensor_face_imgs to (batch_size, channels, height, width) format
        tensor_face_imgs = [img.permute(1,2,0) for img in tensor_face_imgs]

        # Returning both PIL images and tensors for faces
        return (face_imgs, tensor_face_imgs)
