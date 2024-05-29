import copy
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageDraw
from ..flashface.all_finetune.utils import PadToSquare, get_padding
from ..ldm.models.retinaface import crop_face, retinaface

# Initialize padding and transformations
padding_to_square = PadToSquare(224)
retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])

# Load pretrained RetinaFace model
retinaface = retinaface(pretrained=True, device='cuda').eval().requires_grad_(False)

class FlashFaceDetectFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PIL_IMAGE", "IMAGE")
    FUNCTION = "detect_face"
    CATEGORY = "FlashFace"

    def detect_face(self, **kwargs):
        images = kwargs.get('images')
        
        # Validate input is a 4D tensor
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError("Input should be a 4D tensor of images")

        pil_imgs = []
        tensor_imgs = []

        # Convert each tensor image to PIL and store both formats
        for img in images:
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            pil_image = F.to_pil_image(img)
            pil_imgs.append(pil_image)
            tensor_imgs.append(img)

        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # Transform images for RetinaFace and move to CUDA device
        imgs = torch.stack([retinaface_transforms(u) for u in pil_imgs]).to('cuda')
        boxes, kpts = retinaface.detect(imgs, min_thr=0.6)

        face_imgs = []
        tensor_face_imgs = []

        for i in range(b):
            # Scale and padding adjustment
            scale = 640 / max(pil_imgs[i].size)
            left, top, _, _ = get_padding(round(scale * pil_imgs[i].width),
                                          round(scale * pil_imgs[i].height), 640)

            boxes[i][:, [0, 2]] -= left
            boxes[i][:, [1, 3]] -= top
            kpts[i][:, :, 0] -= left
            kpts[i][:, :, 1] -= top

            boxes[i][:, :4] /= scale
            kpts[i][:, :, :2] /= scale

            # Crop face(s) from the image
            crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
            if len(crops) != 1:
                raise ValueError(f'Found {len(crops)} faces in the image {i + 1}, please ensure there is only one face in each image')

            # Resize cropped face to 224x224 and store both PIL and tensor formats
            for crop in crops:
                resized_crop = crop.resize((224, 224))
                face_imgs.append(resized_crop)
                tensor_face_imgs.append(F.to_tensor(resized_crop).unsqueeze(0))

            # Draw bounding boxes on the visualization images
            draw = ImageDraw.Draw(vis_pil_imgs[i])
            for box in boxes[i]:
                box = box[:4].tolist()
                box = [int(x) for x in box]
                draw.rectangle(box, outline='red', width=4)

        # Stack tensors into a 4D tensor
        if tensor_face_imgs:
            tensor_face_imgs = torch.cat(tensor_face_imgs, dim=0)

        # Permute tensor_face_imgs to (batch_size, height, width, channels)
        tensor_face_imgs = [img.permute(1,2,0) for img in tensor_face_imgs]

        # Return both PIL images and the corresponding 4D tensor
        return (face_imgs, tensor_face_imgs)
