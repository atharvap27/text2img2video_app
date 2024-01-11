from flask import Flask, request, jsonify
import torch
from PIL import Image
from io import BytesIO
import base64
from diffusers import StableDiffusionImg2ImgPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

app = Flask(__name__)

@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.json
    prompt = data['prompt']
    image_data = data['image']
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Initialize and run img2img pipeline
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe_img2img.enable_attention_slicing()
    
    gen_images = pipe_img2img(prompt=prompt, num_images_per_prompt=1, image=image, strength=0.8, num_inference_steps=80, guidance_scale=15)
    output_image = gen_images.images[0]

    # Initialize and run video pipeline
    pipe_video = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe_video.enable_model_cpu_offload()
    
    generator = torch.manual_seed(42)
    frames = pipe_video(output_image.resize((1024, 576)), decode_chunk_size=1, generator=generator).frames[0]

    video_path = "/path/to/save/generated.mp4"  # Update the path as needed
    export_to_video(frames, video_path, fps=7)

    # Return video path or video as response (depending on your setup)
    return jsonify({"video_path": video_path})

if __name__ == '__main__':
    app.run(port=5000)
