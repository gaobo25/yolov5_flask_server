import argparse
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import Flask, request, jsonify
from utils.YoloBody import YOLO
from utils.configs import *


def get_picture_base64_data(image_path):
    with open(image_path, "rb") as img_obj:
        base64_data = base64.b64encode(img_obj.read())
        # base64_str = str(base64_data, 'utf-8')
        base64_str = base64_data.decode('utf-8')
    return base64_str


def base_to_cv2(b64str):
    try:
        datas = base64.b64decode(b64str.encode('utf-8'))
        datas = np.frombuffer(datas, np.uint8)
        image = cv2.imdecode(datas, cv2.IMREAD_COLOR)
        if image is not None:
            return image
        else:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise ValueError("Error decoding image: " + str(e))


class YOLOSystem():
    def __init__(self):
        self.detect = YOLO(onnx_path=model_path, device=device)

    def predict(self, images):
        all_results = []
        for img in images:
            if img is None:
                all_results.append([])
                continue
            res = self.detect.inferenced(img)
            all_results.append(res)
        return all_results

    def serving_method(self, images, **kwargs):
        """
        run as a server
        """
        if isinstance(images, str):
            images = [images]
        if not images:
            return []
        images_decode = [base_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    app = Flask(__name__)
    net = YOLOSystem()


    @app.route("/detect", methods=["POST"])
    def recognition():
        try:
            data = request.get_json()
            file_img = data.get('images', [])
            image_data = [get_picture_base64_data(img_path) for img_path in file_img]
            out = net.serving_method(image_data)
            res = {'results': out}

            logging.debug(f"Recognition results: {res}")
            return jsonify(res)

        except Exception as e:

            logging.error(f"Error during recognition: {str(e)}")
            return jsonify({'error': 'Internal Server Error'}), 500


    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO)

    executor = ThreadPoolExecutor()

    app.run(host="0.0.0.0", port=args.port, debug=False)
