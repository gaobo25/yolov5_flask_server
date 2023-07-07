import os

import cv2
import numpy as np
import onnxruntime as ort

from .configs import *
from .postprocessing import non_max_suppression, tag_images, plot_QUAD


class ONNXModel(object):
    def __init__(self, onnx_path, device):
        """

        :param onnx_path:
        :param device: CPU or GPU
        """
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_feed(self, input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed


class YOLO(ONNXModel):
    def __init__(self, onnx_path=model_path, device=device):
        super(YOLO, self).__init__(onnx_path, device)
        # 训练时的图片大小
        self.img_size = 640
        self.img_size_h = self.img_size_w = self.img_size
        self.batch_size = 1

        self.classes = CLASSES

    def to_numpy(self, file, shape, gray=False):
        def letterbox_image(image, size):
            if img is None:
                raise ValueError("Input image is None.")
            iw, ih = image.shape[1], image.shape[0]
            w, h = size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.ones((h, w, 3), dtype=np.uint8) * 128
            new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = image
            return new_image

        if isinstance(file, np.ndarray):
            img = file
        elif isinstance(file, bytes):
            nparr = np.frombuffer(file, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(file)

        resized = letterbox_image(img, shape)
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        return np.expand_dims(img_in, axis=0)

    def inferenced(self, file):
        image_numpy = self.to_numpy(file, shape=(self.img_size, self.img_size))
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        pred = non_max_suppression(outputs[0])
        image = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
        if pred:
            res = tag_images(image, pred, self.img_size, self.classes)
        else:
            res = []
        return res


if __name__ == '__main__':
    image_path = "../data"
    model_path = "../models/yolov5s.onnx"
    for img in os.listdir(image_path):
        new_img = os.path.join(image_path, img)
        model = YOLO(onnx_path=model_path, device="cpu")
        new_img = cv2.imread(new_img)
        res = model.inferenced(new_img)

        image = plot_QUAD(new_img, res)
        # cv2.imshow("test" + img, image)
        # cv2.waitKey(0)
