import json
from typing import Dict


class Config:
    def __init__(self):
        # 物品配置
        self.object_configs = {
            "苹果": {
                "name": "苹果",
                "texts": ['apple', 'apples'],
                "trigger_prob_threshold": 0.95,
                "summarize_frames": 25,
                "summarize_prob": 0.95,
                "frame_proportion": 0.8
            },
        }

        # 字体文件
        self.font_file = r"D:\resources\Fonts\微软雅黑.ttf"
        # 识别设备，cpu or cuda
        self.device = "cuda"
        # CLIP 模型所在目录
        self.model_directory = r"D:\models"

        # 最高帧率
        self.fps = 25
        # 后端地址
        self.notify_url = "http://localhost:9999/update_display"
        self.enable_network_notify = True
        # 视频流地址
        self.video_stream = 1

        # 所有texts
        self.texts = []
        # 索引
        self.text2name = dict()

    def get_text_trigger_condition(self, text: str):
        name = self.text2name[text]
        return self.object_configs[name]

    def parse(self, file: str):
        data: Dict[str, Dict] = json.load(open(file, "r", encoding="utf-8"))
        self.__dict__.update(data)

        if not data.get("object_configs"):
            raise Exception("object_configs is null or empty.")

        obj_cfgs = data.get("object_configs")
        for obj in obj_cfgs.values():
            if not obj.get("texts"):
                continue
            for text in obj.get("texts"):
                self.text2name[text] = obj.get("name")
            self.texts.extend(obj.get("texts"))

        return self

    def to_dict(self):
        return self.__dict__


if __name__ == '__main__':
    cfg = Config().parse("config.json")
    print(cfg.to_dict())
