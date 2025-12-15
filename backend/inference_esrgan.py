"""
esr_gan.py

- Loads your trained diffusion UNet from model_epoch_449.pt
- Generates low-res 128x128 anime images.
- Uses a pretrained Real-ESRGAN x4 anime model to upscale to ~512x512.
- Saves:
    - lr_samples.png         (diffusion outputs, 128x128)
    - sr_samples.png         (ESRGAN super-res, ~512x512)
"""

import os
import logging
import sys
import types

import torch
from torchvision import utils as vutils
from diffusers import UNet2DModel, DDPMScheduler
import torchvision.transforms.functional as F
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

DIFFUSION_CKPT = "model_epoch_449.pt"

OUTPUT_DIR = "./anime_diffusion_sr_outputs"

IMAGE_SIZE = 128
UPSCALE = 4
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 1000
NUM_SAMPLES = 1
SEED = None
ESRGAN_WEIGHTS = "realesrgan_weights/RealESRGAN_x4plus_anime_6B.pth"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "sample_with_esrgan.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_diffusion_model(device, checkpoint_path: str = None, image_size: int = None, num_train_timesteps: int = None):
    logger = logging.getLogger(__name__)
    
    # Use provided parameters or fall back to global constants
    ckpt_path = checkpoint_path if checkpoint_path is not None else DIFFUSION_CKPT
    img_size = image_size if image_size is not None else IMAGE_SIZE
    timesteps = num_train_timesteps if num_train_timesteps is not None else NUM_TRAIN_TIMESTEPS

    model = UNet2DModel(
        sample_size=img_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="linear"
    )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Diffusion checkpoint not found: {ckpt_path}")

    logger.info(f"Loading diffusion checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, noise_scheduler


@torch.no_grad()
def sample_diffusion(
    model,
    noise_scheduler,
    device,
    num_samples=1,
    image_size=128,
    num_inference_steps=1000,
):
    logger = logging.getLogger(__name__)
    logger.info(
        f"Sampling {num_samples} images with {num_inference_steps} diffusion steps..."
    )

    model.eval()

    noise_scheduler.set_timesteps(num_inference_steps)

    x = torch.randn(num_samples, 3, image_size, image_size, device=device)

    for t in noise_scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch).sample
        x = noise_scheduler.step(noise_pred, t, x).prev_sample

    x = (x.clamp(-1, 1) + 1) / 2.0
    logger.info("Sampling finished.")
    return x


# =========================
# ESRGAN: LOAD + UPSCALE
# =========================

def load_esrgan(device, weights_path: str, scale: int = 4):
    logger = logging.getLogger(__name__)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"ESRGAN weights not found: {weights_path}\n"
            "Download RealESRGAN_x4plus_anime_6B.pth and update ESRGAN_WEIGHTS."
        )

    logger.info(f"Loading ESRGAN model from {weights_path}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=scale
    )

    half_precision = (device.type == "cuda")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=weights_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half_precision
    )

    logger.info("ESRGAN model loaded.")
    return upsampler


def torch_batch_to_esrgan(
    lr_batch: torch.Tensor,
    upsampler: RealESRGANer
) -> torch.Tensor:
    logger = logging.getLogger(__name__)
    logger.info("Upscaling batch with ESRGAN...")

    lr_batch = lr_batch.detach().cpu()
    sr_imgs = []

    for idx in range(lr_batch.size(0)):
        img = lr_batch[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

        img_bgr = img_np[:, :, ::-1]

        sr_bgr, _ = upsampler.enhance(img_bgr, outscale=upsampler.scale)
        sr_rgb = sr_bgr[:, :, ::-1]

        sr_rgb = sr_rgb.astype(np.float32) / 255.0
        sr_tensor = torch.from_numpy(sr_rgb).permute(2, 0, 1)
        sr_imgs.append(sr_tensor)

    sr_batch = torch.stack(sr_imgs, dim=0)
    logger.info("ESRGAN upscaling finished.")
    return sr_batch


# =========================
# API FUNCTION
# =========================

def generate_esrgan_image(
    checkpoint_path: str = "model_epoch_449.pt",
    esrgan_weights_path: str = "realesrgan_weights/RealESRGAN_x4plus_anime_6B.pth",
    output_dir: str = "./anime_diffusion_sr_outputs",
    num_samples: int = 1,
    image_size: int = 128,
    seed: int = None,
    device_str: str = "cuda",
    num_inference_steps: int = 1000,
    scale: int = 4,
):
    import time
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)
    
    if seed is None:
        import random
        random_seed = random.randint(0, 2**32 - 1)
        set_seed(random_seed)
        logger.info(f"Using random seed: {random_seed}")
    else:
        set_seed(seed)
        logger.info(f"Using fixed seed: {seed}")
    
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    diffusion_model, noise_scheduler = load_diffusion_model(
        device, 
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        num_train_timesteps=1000
    )
    
    lr_imgs = sample_diffusion(
        diffusion_model,
        noise_scheduler,
        device,
        num_samples=num_samples,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
    )
    
    lr_save_path = os.path.join(output_dir, "lr_samples.png")
    vutils.save_image(
        lr_imgs,
        lr_save_path,
        nrow=int(num_samples ** 0.5)
    )
    logger.info(f"Saved low-res diffusion samples to {lr_save_path}")
    
    upsampler = load_esrgan(device, esrgan_weights_path, scale=scale)
    
    sr_imgs = torch_batch_to_esrgan(lr_imgs, upsampler)
    
    sr_save_path = os.path.join(output_dir, "sr_samples.png")
    vutils.save_image(
        sr_imgs,
        sr_save_path,
        nrow=int(num_samples ** 0.5)
    )
    logger.info(f"Saved super-res ESRGAN samples to {sr_save_path}")
    
    processing_time = time.time() - start_time
    logger.info(f"Done. Processing time: {processing_time:.2f}s")
    
    return lr_save_path, sr_save_path, processing_time


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger(OUTPUT_DIR)
    
    if SEED is None:
        import random
        random_seed = random.randint(0, 2**32 - 1)
        set_seed(random_seed)
        logger.info(f"Using random seed: {random_seed}")
    else:
        set_seed(SEED)
        logger.info(f"Using fixed seed: {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    diffusion_model, noise_scheduler = load_diffusion_model(device, checkpoint_path=DIFFUSION_CKPT, image_size=IMAGE_SIZE, num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    lr_imgs = sample_diffusion(
        diffusion_model,
        noise_scheduler,
        device,
        num_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    lr_save_path = os.path.join(OUTPUT_DIR, "lr_samples.png")
    vutils.save_image(
        lr_imgs,
        lr_save_path,
        nrow=int(NUM_SAMPLES ** 0.5)
    )
    logger.info(f"Saved low-res diffusion samples to {lr_save_path}")

    upsampler = load_esrgan(device, ESRGAN_WEIGHTS, scale=UPSCALE)

    sr_imgs = torch_batch_to_esrgan(lr_imgs, upsampler)

    sr_save_path = os.path.join(OUTPUT_DIR, "sr_samples.png")
    vutils.save_image(
        sr_imgs,
        sr_save_path,
        nrow=int(NUM_SAMPLES ** 0.5)
    )
    logger.info(f"Saved super-res ESRGAN samples to {sr_save_path}")

    logger.info("Done. Check lr_samples.png vs sr_samples.png.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("esr_gan.py crashed with an exception")
        raise