import base64

from flask import Flask, request, jsonify
import os
import json
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
import io
import torch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10240 * 1024 * 1024
# 定義上傳目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    print(1)
    data = request.get_json()
    print(2)
    if not data or 'image' not in data:
        return jsonify({"message": "No image data in request"}), 400

    image_data = data['image']
    prompt_user = data['prompt']
    try:
        # 解碼 Base64 圖片數據
        decoded_image = base64.b64decode(image_data)
        # 儲存圖片
        image_path = os.path.join('uploads', 'uploaded_image.jpg')
        with open(image_path, 'wb') as image_file:
            image_file.write(decoded_image)

        # model path
        #SDV5_MODEL_PATH = 'F:/stable diffusion/stable-diffusion-v1-5'
        APD_MODEL_PATH = 'F:/stable diffusion/anime-pastel/anime-pastel-dream-soft-baked-vae'
        SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')

        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        def uniquify(path):
            filename, extension = os.path.splitext(path)
            counter = 1

            while os.path.exists(path):
                path = filename + ' (' + str(counter) + ')' + extension
                counter += 1

            return path
        # load image
        img_path = 'E:/cs/4(1)/logic/uploads/uploaded_image.jpg'

        init_image = Image.open(open(img_path, 'rb')).convert("RGB")
        init_image = init_image.resize((480, 480))

        # prompt define
        prompt = "anime illustration, detailed beautiful eyes, detailed beautiful face, detailed beautiful hair, best quality, highly detailed, black_eyes, black_hair, short hair, hair_between_eyes, kirito, "+prompt_user
        negative_prompt = "worst quality, ugly, bad anatomy, jpeg artifacts, nsfw, text, watermark, bad hands, extra digit, fewer digits, out of focus, JPEG artifacts, low resolution, nipples, long_body, mutated hands, missing arms, extra_arms, extra_legs, bad hands, missing_limb, disconnected_limbs, extra_fingers, missing fingers, liquid fingers, ugly face, deformed eyes, cropped, text, signature, split view, grid view, two shot, poorly drawn eyes, weird face, strange face"


        # load model
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(APD_MODEL_PATH)
        pipe = pipe.to('cuda')
        pipe.safety_checker = lambda images, clip_input: (images, False)
        #pipe.load_lora_weights(".", weight_name="kirito.safetensors")

        #generate image
        with autocast('cuda'):
            image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

        image_path = uniquify(os.path.join(SAVE_PATH, (prompt[:25] + '...') if len(prompt) > 25 else prompt) + '.png')
        print(image_path)


        image_path = 'transform/test_image.png'
        image.save(image_path)
        base64_string = image_to_base64(image_path)

        return jsonify({"image": base64_string}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500


def image_to_base64(image_path):
    # 打開圖片文件
    with Image.open(image_path) as image:
        # 將圖片轉換為二進制數據
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_data = buffered.getvalue()

        # 將二進制數據轉換為 Base64 編碼
        img_base64 = base64.b64encode(img_data)
        return img_base64.decode('utf-8')

@app.errorhandler(400)
def bad_request(error):
    return "Bad Request: {}".format(error), 400

@app.route('/get', methods=['GET'])
def get_file():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"message": "No image data in request"}), 400

    image_data = data['image']
    try:
        # 解碼 Base64 圖片數據
        decoded_image = base64.b64decode(image_data)
        # 儲存圖片
        image_path = os.path.join('uploads', 'uploaded_image.jpg')
        with open(image_path, 'wb') as image_file:
            image_file.write(decoded_image)




        return jsonify({"Image": image_data}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)