import sys
from queue import Queue
from typing import Optional, Callable
from threading import Thread

from PIL.Image import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

from config import Config
from main_context import MainContext
from summarize_processor import SummarizeProcessor
from frame_processor import FrameProcessor


class QueueListener(Thread):
    def __init__(self, queue: Queue, callback: Callable[[Image], None]):
        super().__init__()
        self.queue: Queue = queue
        self._running = True
        self.callback = callback

    def run(self):
        # 在单独的线程中监听队列
        while self._running:
            # 从队列中获取数据（阻塞）
            self.callback(self.queue.get())

    def stop(self):
        self._running = False


class VideoPlayerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.video_label)

        self.container = QWidget()
        self.container.setLayout(self.vbox)
        self.setCentralWidget(self.container)

        self.frame_queue: Optional[Queue] = None
        # 队列监听线程
        self.frame_updater: Optional[QueueListener] = None

    def enable_frame_updater(self, _q: Queue):
        self.frame_queue = _q

    def update_frame(self, frame: Image):
        if not frame:
            return
        self.video_label.setPixmap(frame.toqpixmap())

    def close(self) -> None:
        super().close()

    def show(self) -> None:
        if self.frame_queue is not None:
            self.frame_updater = QueueListener(self.frame_queue, self.update_frame)
            self.frame_updater.setDaemon(True)
            self.frame_updater.start()

        available_geometry = self.screen().availableGeometry()
        self.resize(int(available_geometry.width() / 3), int(available_geometry.height() / 2))
        super().show()


def setup_test_ui(cfg: Config):
    app = QApplication(sys.argv)
    window = VideoPlayerWindow()
    frame_buffer_queue: Queue = Queue(maxsize=30)

    main_context = MainContext(cfg)
    main_context.set_processors(
        [SummarizeProcessor(cfg, cfg.enable_network_notify), FrameProcessor(cfg, frame_buffer_queue)])
    main_context.setDaemon(True)
    main_context.start()

    window.enable_frame_updater(frame_buffer_queue)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    setup_test_ui(Config().parse("config.json"))
