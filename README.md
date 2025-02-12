# Yotraco

## Overview

Yotraco is an object tracking and counting system based on YOLO (You Only Look Once). It processes videos to detect, track, and count objects crossing a defined line in specified directions. It supports multiple object classes and can track both "IN" and "OUT" movements.

## Features

- **Object Detection & Tracking**: Uses YOLO to detect and track objects in a video.
- **Crossing Detection**: Counts objects that cross a defined line in the frame.
- **Customizable Settings**:
  - Define tracking line position (top, middle, bottom).
  - Specify tracking direction (IN, OUT, or BOTH).
  - Select specific object classes to track.
- **Processed Video Output**: Saves the processed video with tracking and count overlays.
- **Logging & Statistics**: Maintains statistics of tracked objects.

## Installation

```bash
pip install -r requirements.txt
```

Ensure you have `ultralytics` installed for YOLO model support.

```bash
pip install ultralytics
```

## Usage

```python
from yotraco import Yotraco

yotraco = Yotraco(
    model_path='path/to/yolo_model.pt',
    video_path='path/to/input_video.mp4',
    output_video='path/to/output_video.avi',
    line_position='middle',
    track_direction='BOTH',
    classes_to_track=[0, 1, 2, 3]
)
yotraco.process_video()
```

## Dependencies

- Python 3.7+
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- NumPy
- 

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Authors

YOTRACO TEAM : Ben Yamna Mohammed , Makkour Israe 
