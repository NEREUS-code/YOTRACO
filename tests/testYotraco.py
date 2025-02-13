import unittest
from YOTRACO.yotraco import Yotraco
from PIL import Image
import numpy as np



class TestYotraco(unittest.TestCase):

    def setUp(self):
        """
        Set up mock objects and the environment before each test.
        """
        self.model_path = "tests/model/yolo11l.pt"
        self.video_path = "tests/videos/test_video.mp4"
        self.output_video = "output/video.avi"
        self.yotraco = Yotraco(self.model_path, self.video_path, self.output_video)

    def test_model_loading(self):
        """
        Test if the YOLO model is loaded correctly
        """
        self.assertTrue(self.yotraco.model is not None)
        # TODO : fix the model equality test
        # self.assertEqual(self.yotraco.model.model, self.model_path)


    def test_video_loading(self):
        """Test if the video is opened correctly"""
        self.assertTrue(self.yotraco.cap.isOpened())

    def test_process_frame(self):
        """Test the frame processing functionality"""
        frame = Image.new('RGB', (640, 480), color = (73, 109, 137))  # Create a real image
        frame = np.array(frame)
        self.yotraco.process_frame(frame)


if __name__ == "__main__":
    unittest.main()
