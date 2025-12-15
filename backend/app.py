import os
import sys
import time
import importlib.util
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

try:
    from inference_diffusion import generate_diffusion_image
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "inference_diffusion", 
        os.path.join(os.path.dirname(__file__), "inference_diffusion.py")
    )
    inference_diffusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_diffusion)
    generate_diffusion_image = inference_diffusion.generate_diffusion_image

try:
    from inference_esrgan import generate_esrgan_image
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "inference_esrgan", 
        os.path.join(os.path.dirname(__file__), "inference_esrgan.py")
    )
    inference_esrgan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_esrgan)
    generate_esrgan_image = inference_esrgan.generate_esrgan_image

try:
    from NST_inference import generate_nst_image
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "NST_inference", 
        os.path.join(os.path.dirname(__file__), "NST_inference.py")
    )
    nst_inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nst_inference)
    generate_nst_image = nst_inference.generate_nst_image

app = Flask(__name__)
CORS(app)


@app.route('/')
def root():
    return jsonify({
        "message": "Neural Canvas API - Diffusion Lab",
        "version": "1.0.0",
        "endpoints": {
            "/api/generate-diffusion": "POST - Generate image using diffusion model",
            "/api/generate-esrgan": "POST - Generate image with diffusion + ESRGAN upscaling",
            "/api/transfer-style": "POST - Neural Style Transfer",
            "/api/generated-images/<filename>": "GET - Serve generated images",
            "/api/esrgan-images/<filename>": "GET - Serve ESRGAN generated images",
            "/api/nst-images/<filename>": "GET - Serve NST generated images",
            "/api/health": "GET - Health check"
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })


@app.route('/api/generate-diffusion', methods=['POST'])
def generate_diffusion():
    import sys
    import traceback
    
    try:
        data = request.json or {}
        num_samples = data.get('num_samples', 1)
        inference_steps = data.get('inference_steps', None)
        seed = data.get('seed', None)
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(backend_dir, "model_epoch_449.pt")
        output_dir = os.path.join(backend_dir, "anime_inference_outputs")
        
        try:
            output_path, processing_time = generate_diffusion_image(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                num_samples=num_samples,
                image_size=128,
                seed=seed,
                device_str=device_str,
                num_inference_steps=inference_steps,
            )
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                error_msg = f"GPU out of memory. Try reducing num_samples or inference_steps. Original error: {error_msg}"
            print(f"ERROR in generate_diffusion_image: {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return jsonify({"error": error_msg}), 500
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU out of memory. Try reducing num_samples or inference_steps. Original error: {str(e)}"
            print(f"ERROR (CUDA OOM): {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return jsonify({"error": error_msg}), 500
        
        image_url = f"http://localhost:5000/api/generated-images/generated_samples.png"
        
        return jsonify({
            "imageUrl": image_url,
            "processingTime": int(processing_time * 1000)  # Convert to milliseconds
        })
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(f"ERROR in generate_diffusion endpoint: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": error_msg}), 500


@app.route('/api/generate-esrgan', methods=['POST'])
def generate_esrgan():
    try:
        data = request.json or {}
        num_samples = data.get('num_samples', 1)
        inference_steps = data.get('inference_steps', 1000)
        seed = data.get('seed', None)
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(backend_dir, "model_epoch_449.pt")
        esrgan_weights_path = os.path.join(backend_dir, "realesrgan_weights/RealESRGAN_x4plus_anime_6B.pth")
        output_dir = os.path.join(backend_dir, "anime_diffusion_sr_outputs")
        
        lr_path, sr_path, processing_time = generate_esrgan_image(
            checkpoint_path=checkpoint_path,
            esrgan_weights_path=esrgan_weights_path,
            output_dir=output_dir,
            num_samples=num_samples,
            image_size=128,
            seed=seed,
            device_str=device_str,
            num_inference_steps=inference_steps,
            scale=4,
        )
        
        lr_filename = os.path.basename(lr_path)
        sr_filename = os.path.basename(sr_path)
        
        lr_image_url = f"http://localhost:5000/api/esrgan-images/{lr_filename}"
        sr_image_url = f"http://localhost:5000/api/esrgan-images/{sr_filename}"
        
        return jsonify({
            "lrImageUrl": lr_image_url,
            "srImageUrl": sr_image_url,
            "processingTime": int(processing_time * 1000)  # Convert to milliseconds
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating ESRGAN image: {str(e)}"}), 500


@app.route('/api/generated-images/<filename>')
def serve_generated_image(filename):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(backend_dir, "anime_inference_outputs")
    return send_from_directory(output_dir, filename)


@app.route('/api/esrgan-images/<filename>')
def serve_esrgan_image(filename):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(backend_dir, "anime_diffusion_sr_outputs")
    return send_from_directory(output_dir, filename)


@app.route('/api/transfer-style', methods=['POST'])
def transfer_style():
    import base64
    import uuid
    import traceback
    
    try:
        data = request.json or {}
        style_image_data = data.get('styleImage')
        content_image_data = data.get('contentImage')
        
        if not style_image_data or not content_image_data:
            return jsonify({"error": "Both styleImage and contentImage are required"}), 400
        
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(backend_dir, "nst_temp")
        output_dir = os.path.join(backend_dir, "nst_outputs")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        unique_id = str(uuid.uuid4())[:8]
        style_image_path = os.path.join(temp_dir, f"style_{unique_id}.jpg")
        content_image_path = os.path.join(temp_dir, f"content_{unique_id}.jpg")
        
        if style_image_data.startswith('data:image'):
            header, encoded = style_image_data.split(',', 1)
            image_data = base64.b64decode(encoded)
        else:
            image_data = base64.b64decode(style_image_data)
        
        with open(style_image_path, 'wb') as f:
            f.write(image_data)
        
        if content_image_data.startswith('data:image'):
            header, encoded = content_image_data.split(',', 1)
            image_data = base64.b64decode(encoded)
        else:
            image_data = base64.b64decode(content_image_data)
        
        with open(content_image_path, 'wb') as f:
            f.write(image_data)
        
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint_path = os.path.join(backend_dir, "epoch_46.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(backend_dir, "best.pt")
        
        style_path, content_path, generated_path, processing_time = generate_nst_image(
            content_image_path=content_image_path,
            style_image_path=style_image_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            size=256,
            device_str=device_str,
        )
        
        style_filename = os.path.basename(style_path)
        content_filename = os.path.basename(content_path)
        generated_filename = os.path.basename(generated_path)
        
        style_image_url = f"http://localhost:5000/api/nst-images/{style_filename}"
        content_image_url = f"http://localhost:5000/api/nst-images/{content_filename}"
        generated_image_url = f"http://localhost:5000/api/nst-images/{generated_filename}"
        
        return jsonify({
            "styleImageUrl": style_image_url,
            "contentImageUrl": content_image_url,
            "imageUrl": generated_image_url,
            "processingTime": int(processing_time * 1000)  # Convert to milliseconds
        })
        
    except Exception as e:
        error_msg = f"Error in style transfer: {str(e)}"
        print(f"ERROR in transfer_style endpoint: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": error_msg}), 500


@app.route('/api/nst-images/<filename>')
def serve_nst_image(filename):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(backend_dir, "nst_outputs")
    if os.path.exists(os.path.join(output_dir, filename)):
        return send_from_directory(output_dir, filename)
    
    temp_dir = os.path.join(backend_dir, "nst_temp")
    if os.path.exists(os.path.join(temp_dir, filename)):
        return send_from_directory(temp_dir, filename)
    
    return jsonify({"error": "Image not found"}), 404


if __name__ == '__main__':
    import sys
    import atexit
    
    def cleanup():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    atexit.register(cleanup)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print("=" * 80, file=sys.stderr)
        print("UNCAUGHT EXCEPTION - Flask app crashed", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
    
    sys.excepthook = handle_exception
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\nShutting down Flask server...")
        cleanup()
    except Exception as e:
        print(f"Fatal error starting Flask server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        cleanup()
        raise
