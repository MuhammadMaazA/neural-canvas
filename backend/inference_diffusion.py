import os
import argparse
import logging
import random
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torchvision import utils as vutils
from diffusers import UNet2DModel, DDPMScheduler


NUM_TRAIN_TIMESTEPS = 1000
IMAGE_SIZE = 128
SEED = None


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, device):
    logger = logging.getLogger(__name__)
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
    
    try:
        logger.info("Creating UNet2DModel architecture...")
        model = UNet2DModel(
            sample_size=IMAGE_SIZE,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        logger.info("Moving model to device...")
        model = model.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA OOM while creating/moving model: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Out of memory while loading model: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise
    
    try:
        logger.info("Loading model state dict...")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Model loaded successfully. Trained for {ckpt.get('epoch', 'unknown')} epochs.")
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        raise
    
    return model


def generate_samples(
    model,
    noise_scheduler,
    device,
    num_samples: int = 1,
    image_size: int = 128,
    num_inference_steps: int = 1000,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_samples} samples with {num_inference_steps} inference steps...")
    
    try:
        # Set timesteps for faster inference (reduces from 1000 to num_inference_steps)
        noise_scheduler.set_timesteps(num_inference_steps)
        timesteps = noise_scheduler.timesteps
        
        model.eval()
        with torch.no_grad():
            # Start from Gaussian noise
            logger.info(f"Creating initial noise tensor on {device}...")
            x = torch.randn(num_samples, 3, image_size, image_size, device=device)
            logger.info(f"Initial noise tensor created. Shape: {x.shape}")
            
            # Use tqdm for progress bar
            try:
                for i, t in enumerate(tqdm(timesteps, desc="Generating", unit="step")):
                    try:
                        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                        noise_pred = model(x, t_batch).sample  # predict noise
                        
                        step_output = noise_scheduler.step(noise_pred, t, x)
                        x = step_output.prev_sample
                        
                        # Log progress every 100 steps
                        if (i + 1) % 100 == 0:
                            logger.info(f"Completed {i + 1}/{len(timesteps)} steps")
                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"CUDA OOM error at step {i+1}/{len(timesteps)}: {e}")
                        raise
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error(f"CUDA OOM error at step {i+1}/{len(timesteps)}: {e}")
                            raise
                        else:
                            logger.error(f"Runtime error at step {i+1}/{len(timesteps)}: {e}")
                            raise
            except KeyboardInterrupt:
                logger.warning("Generation interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Error during generation loop: {type(e).__name__}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            # Denormalize from [-1,1] to [0,1]
            logger.info("Denormalizing output...")
            x = (x.clamp(-1, 1) + 1) / 2.0
        
        logger.info("Generation complete!")
        return x
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory error: {e}")
        logger.error("Try reducing num_samples, image_size, or num_inference_steps")
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA Out of Memory error: {e}")
            logger.error("Try reducing num_samples, image_size, or num_inference_steps")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_samples: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def save_grid(samples, output_path: str, nrow: int = None):
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if nrow is None:
        nrow = int(samples.size(0) ** 0.5)
    
    vutils.save_image(samples, output_path, nrow=nrow)
    logger.info(f"Saved grid image to {output_path}")


def generate_diffusion_image(
    checkpoint_path: str = "model_epoch_449.pt",
    output_dir: str = "./anime_inference_outputs",
    num_samples: int = 1,
    image_size: int = 128,
    seed: int = None,
    device_str: str = "cuda",
    num_inference_steps: int = None,
):
    import time
    start_time = time.time()
    
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
    
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    if num_inference_steps is None:
        num_inference_steps = 20 if device.type == "cpu" else 1000
        logger.info(f"Using default inference steps: {num_inference_steps} (optimized for {device.type})")
    
    logger.info(f"Using device: {device}")
    
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")
    
    set_seed(seed)
    
    model = load_model(checkpoint_path, device)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="linear"
    )
    logger.info("DDPMScheduler created.")
    
    try:
        samples = generate_samples(
            model,
            noise_scheduler,
            device,
            num_samples=num_samples,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA Out of Memory during generation: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(f"Out of memory error: {e}. Try reducing num_samples or num_inference_steps.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"Out of Memory error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Out of memory error: {e}. Try reducing num_samples or num_inference_steps.")
        raise
    except Exception as e:
        logger.error(f"Error during sample generation: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "generated_samples.png")
        save_grid(samples, output_path)
    except Exception as e:
        logger.error(f"Error saving output: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    processing_time = time.time() - start_time
    logger.info(f"Inference complete! Results saved to {output_dir} (took {processing_time:.2f}s)")
    
    return output_path, processing_time


def main():
    parser = argparse.ArgumentParser(
        description="Generate anime images using a trained DDPM diffusion model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_epoch_449.pt",
        help="Path to the trained model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./anime_inference_outputs",
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size (must match training)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: None = random seed each run)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Number of inference steps (fewer = faster, default: 20 for CPU,  for CUDA)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    if args.inference_steps is None:
        args.inference_steps = 20 if device.type == "cpu" else 1000
        logger.info(f"Using default inference steps: {args.inference_steps} (optimized for {device.type})")
    
    logger.info(f"Using device: {device}")
    
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {args.seed}")
    else:
        logger.info(f"Using provided seed: {args.seed}")
    
    set_seed(args.seed)
    
    model = load_model(args.checkpoint, device)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_schedule="linear"
    )
    logger.info("DDPMScheduler created.")
    
    samples = generate_samples(
        model,
        noise_scheduler,
        device,
        num_samples=args.num_samples,
        image_size=args.image_size,
        num_inference_steps=args.inference_steps,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "generated_samples.png")
    save_grid(samples, output_path)
    
    logger.info(f"Inference complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
