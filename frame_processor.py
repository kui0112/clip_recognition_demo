from queue import Queue
from typing import List, Tuple, Optional

from PIL import ImageFont, Image, ImageDraw

from config import Config
from processor import Processor


class FrameProcessor(Processor):
    def __init__(self, cfg: Config, q: Queue):
        self.cfg = cfg
        self.buffer = q
        self.font_file = self.cfg.font_file
        self.font: Optional[ImageFont] = None

    def initialize(self):
        self.font = ImageFont.truetype(font=self.font_file, size=40)

    def process(self, image: Image.Image, res: List[Tuple]):
        # 文字写到图像上
        drawn_image = self.draw_image(res, image)
        # 图像存放到缓冲区
        self.save_image(drawn_image)

    def save_image(self, image: Image.Image):
        if not self.buffer.full():
            self.buffer.put(image)
        else:
            print("WARNING: buffer queue is full, frame has been dropped.")

    def draw_image(self, _values: List[Tuple[str, float]], image: Image.Image):
        draw = ImageDraw.Draw(image)
        for i, (key, value) in enumerate(_values):
            i: int
            value = float(value)
            content = f"{str.split(key, ',')[0]}: {value * 100:.2f}%"
            x = 10
            y = i * 50 + 10
            draw.text((x, y), content, fill=(0, 0, 0), font=self.font)
        return image
