import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from PIL import Image
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
    
    def forward(self, x):
        residual = x  
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out
    

class ImageTransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.bn1   = nn.InstanceNorm2d(32,  affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.InstanceNorm2d(64,  affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.InstanceNorm2d(128, affine=True)
        
        self.residualblocks = nn.Sequential(
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            ResidualBlock(128), ResidualBlock(128),
        )
        
        self.conv4 = nn.ConvTranspose2d(128, 64,  kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4   = nn.InstanceNorm2d(64,  affine=True)
        self.conv5 = nn.ConvTranspose2d(64,  32,  kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5   = nn.InstanceNorm2d(32,  affine=True)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.residualblocks(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return x


def load_checkpoint(checkpoint_path, device):
    import sys
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"Loading checkpoint from {checkpoint_path}...")
    print(f"Checkpoint file size: {file_size} bytes")
    
    if file_size < 1000:
        raise RuntimeError(f"Checkpoint file appears to be too small ({file_size} bytes). File may be corrupted or incomplete.")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except RuntimeError as e:
        if "failed finding central directory" in str(e) or "zip archive" in str(e).lower():
            error_msg = (
                f"Checkpoint file appears to be corrupted or incomplete: {checkpoint_path}\n"
                f"The file's ZIP archive structure is invalid (missing central directory).\n"
                f"This usually means the file was truncated during download or save.\n"
                f"Please re-download or re-save the checkpoint file.\n"
                f"File size: {file_size} bytes"
            )
            print(f"ERROR: {error_msg}", file=sys.stderr)
            raise RuntimeError(error_msg) from e
        else:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            except Exception as e2:
                error_msg = f"Failed to load checkpoint from {checkpoint_path}. Error: {str(e2)}"
                print(f"ERROR: {error_msg}", file=sys.stderr)
                raise RuntimeError(error_msg) from e2
    except Exception as e:
        error_msg = f"Failed to load checkpoint from {checkpoint_path}. Error: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    
    model = ImageTransformNet().to(device)
    
    try:
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        error_msg = f"Failed to load model state dict. Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}. Error: {str(e)}"
        print(f"ERROR: {error_msg}", file=sys.stderr)
        raise RuntimeError(error_msg) from e
    
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Average loss: {checkpoint.get('avg_loss', 'unknown'):.4f}")
    
    if "hparams" in checkpoint:
        hparams = checkpoint["hparams"]
        print(f"Hyperparameters:")
        print(f"  - Learning rate: {hparams.get('lr', 'unknown')}")
        print(f"  - Content weight: {hparams.get('content_w', 'unknown')}")
        print(f"  - Style weight: {hparams.get('style_w', 'unknown')}")
        print(f"  - TV weight: {hparams.get('tv_w', 'unknown')}")
        print(f"  - Style layers: {hparams.get('style_layers', 'unknown')}")
    
    return model, checkpoint.get("hparams", {})


def preprocess_image(image_path, size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    # Resize to model input size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, original_size


def stylize_image(model, image_path, output_path=None, size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Stylizing image: {image_path}")
    
    img_tensor, original_size = preprocess_image(image_path, size, device)
    
    with torch.no_grad():
        stylized_tensor = model(img_tensor)
        stylized_tensor = stylized_tensor.clamp(0, 1).cpu()[0]
    
    to_pil = transforms.ToPILImage()
    stylized_img = to_pil(stylized_tensor)
    
    if original_size != (size, size):
        stylized_img = stylized_img.resize(original_size, Image.LANCZOS)
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_stylized.jpg"
    
    stylized_img.save(output_path)
    print(f"Saved stylized image to: {output_path}")
    
    return stylized_img


def generate_nst_image(
    content_image_path: str,
    style_image_path: str,
    checkpoint_path: str = "best.pt",
    output_dir: str = "./nst_outputs",
    size: int = 256,
    device_str: str = "cuda",
):
    import time
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    current_device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    
    model, hparams = load_checkpoint(checkpoint_path, current_device)
    
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(output_dir, f"nst_result_{unique_id}.jpg")
    
    stylize_image(model, content_image_path, output_path, size, current_device)
    
    processing_time = time.time() - start_time
    print(f"NST inference complete. Processing time: {processing_time:.2f}s")
    
    return style_image_path, content_image_path, output_path, processing_time


def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path to checkpoint file (default: checkpoints/best.pt)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output image (default: auto-generated)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Input image size (default: 256)"
    )
    
    args = parser.parse_args()
    
    model, hparams = load_checkpoint(args.checkpoint, device)
    
    stylize_image(model, args.input, args.output, args.size, device)
    
    print("Inference complete!")


if __name__ == "__main__":
    main()

