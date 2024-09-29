from typing import List, Tuple, Optional
from collections import namedtuple
from threading import Thread

import torch
import clip
from PIL import Image
import cv2

from config import Config
from processor import Processor

FrameRecord = namedtuple("FrameRecord", ["prob", "reach_trigger_condition"])


class PrintProcessor(Processor):

    def initialize(self):
        pass

    def process(self, image: Image.Image, res: List[Tuple[str, float]]):
        print(res)


class MainContext(Thread):
    def __init__(self, _cfg: Config):
        super().__init__()
        # 配置
        self.cfg: Config = _cfg

        # 视频流
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.device = self.cfg.device

        # 预设的label
        self.texts: List[str] = []
        self.tokens = None

        # 模型
        self.model = None
        self.preprocess = None

        # 是否停止
        self.stopped = False

        # 结果处理函数
        self.processors: List[Processor] = []

        self.average_interval = 0

    def set_processors(self, processors: List[Processor]):
        self.processors = processors

    def stop(self):
        self.stopped = True

    def run(self):
        self.texts = self.cfg.texts

        # 连接视频流
        print("load video stream.")
        self.video_capture = cv2.VideoCapture(self.cfg.video_stream)
        # 设置帧率
        self.video_capture.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # 加载模型
        print("load model.")
        if not torch.cuda.is_available():
            raise Exception("cuda environment error.")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=self.cfg.model_directory)
        # 序列化文本
        self.tokens = clip.tokenize(self.texts).to(self.device)
        # 初始化 processors
        for processor in self.processors:
            processor.initialize()

        print("start context.")
        frames = 0
        if not self.video_capture.isOpened():
            print("context was stopped.")
            self.stop()
            return
        try:
            # 主循环
            while self.video_capture.isOpened() and not self.stopped:
                res, frame = self.video_capture.read()
                if not res:
                    break
                # mat to image
                image: Image.Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 图像预处理
                preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    # 计算相似度
                    logits_per_image, logits_per_text = self.model(preprocessed_image, self.tokens)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                # text和识别结果组合
                values: List[Tuple[str, float]] = list(zip(self.texts, probs[0]))

                # 处理结果
                for processor in self.processors:
                    processor.process(image, values)

                frames += 1
        except KeyboardInterrupt as ex:
            print("keyboard interrupt.")

        print("context was stopped.")
        self.stop()


if __name__ == '__main__':
    c = Config().parse("config.json")
    ctx = MainContext(c)
    ctx.set_processors([PrintProcessor()])
    ctx.run()
