import abc
from typing import List, Tuple

from PIL import Image


class Processor(abc.ABC):
    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def process(self, image: Image.Image, res: List[Tuple[str, float]]):
        pass
