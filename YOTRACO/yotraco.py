import os
import cv2
import logging
# import threading
from ultralytics import YOLO
from collections import defaultdict
from YOTRACO.yotracoStats import YotracoStats

logging.basicConfig(level=logging.INFO)  # Set up logging globally

class Yotraco:

    def __init__(self, model_path, video_path, output_video, line_position='middle', track_direction='BOTH', classes_to_track=None , display=True):
        """
        Initialize the YOTRACO object with the specified YOLO model and video processing settings.

        Args:
        model_path (str): Path to the YOLO model file (e.g., a .pt or .onnx file).
        video_path (str): Path to the input video file to be processed.
        output_video (str): Path to save the processed video.
        line_position (str, optional): Vertical position of the tracking line ('top', 'middle', 'bottom'). Default is 'middle'.
        track_direction (str, optional): Direction to track ('BOTH', 'IN', or 'OUT'). Default is 'BOTH'.
        classes_to_track (list, optional): List of class indices to track. Default is [0, 1, 2, 3] (all classes).
        """

        self.stats = YotracoStats() 

        # Check if the model file exists
        if not os.path.exists(model_path):
            logging.info(f"Model file not found at'{model_path}' . \n Downloading...")
        else:
            logging.info(f"Model file found at '{model_path}'. \n Using it...")
            logging.info(f"Loading YOLO model from {model_path}...")

        self.model = YOLO(model_path)  # Load the model
        self.class_list = self.model.names  # Get class names

        logging.info("YOLO model loaded successfully.")

        # Open the video file
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        self.display=display

        # Get video properties
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define output video settings
        self.output_video = output_video + ".avi"
        # TODO : support other extension
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for .avi format
        self.out = cv2.VideoWriter(self.output_video, fourcc, self.fps, (self.frame_width, self.frame_height))

        # TODO : add more control for the lines and add the abillity to put two line vertical and horizontal
        # Set line Y-coordinate based on position , the Y_line is dynamic for the horizantal line
        self.line_position = line_position

        #We need to fix the X_line in the middel for the vertical line
        # self.line_x = int(self._frame_height*0.5)

        # Initialize movement direction and classes to track
        self.track_direction = track_direction
        self.classes_to_track = classes_to_track if classes_to_track is not None else [0, 1, 2, 3]  # Default classes to track

        # Dictionaries for counting IN and OUT events
        self.class_counts_in = defaultdict(int)
        self.class_counts_out = defaultdict(int)
        self.crossed_ids = {}

    # track direction getter
    @property
    def track_direction(self):
        return self._track_direction
    
    # track direction setter
    @track_direction.setter
    def track_direction(self,direction):
        if direction in ['BOTH', 'IN', 'OUT']:
            self._track_direction = direction
        else :
            raise ValueError("track direction must be 'BOTH', 'IN', or 'OUT'")
    
    # line position getter 
    @property
    def line_position(self):
        return self._line_y
    
    # line position setter
    @line_position.setter
    def line_position(self,position):
        if position == 'top':
            self._line_y = int(self.frame_height * 0.3)
        elif position == 'bottom':
            self._line_y = int(self.frame_height * 0.7)
        elif position == 'middle':
            self._line_y = int(self.frame_height * 0.5)  
        else :
            raise ValueError("line position must be 'top', 'middle', 'bottom' ")
            

    def process_frame(self, frame):
        """
        Processes each frame to track objects using the YOLO model, detect crossing, and update counts.

        Args:
        frame (ndarray): A single frame from the video to process.
        """
        # Run YOLO tracking for specified classes
        results = self.model.track(frame, persist=True, classes=self.classes_to_track)

        # Draw tracking line across the full width of the frame
        cv2.line(frame, (0, self._line_y), (self.frame_width, self._line_y), (0, 0, 255), 3)

        # Process detections
        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            if results[0].boxes.id is not None :
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = []
            class_indices = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2  # Calculate the center point of the bounding box
                cy = (y1 + y2) // 2
                class_name = self.class_list[class_idx]

                # Draw bounding box and tracking info
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Track crossing direction
                if track_id not in self.crossed_ids:
                    self.crossed_ids[track_id] = cy
                else:
                    prev_cy = self.crossed_ids[track_id]
                    if self._track_direction == 'BOTH':  # Track both directions
                        if prev_cy < self._line_y <= cy:  # Moving downward (OUT)
                            self.class_counts_out[class_name] += 1
                            self.stats.class_counts_out[class_name] += 1
                        elif prev_cy > self._line_y >= cy:  # Moving upward (IN)
                            self.class_counts_in[class_name] += 1
                            self.stats.class_counts_in[class_name] += 1
                        self.crossed_ids[track_id] = cy
                    elif self._track_direction == 'IN' and prev_cy > self._line_y >= cy:  # Track only IN
                        self.class_counts_in[class_name] += 1
                        self.stats.class_counts_in[class_name] += 1
                        self.crossed_ids[track_id] = cy
                    elif self._track_direction == 'OUT' and prev_cy < self._line_y <= cy:  # Track only OUT
                        self.class_counts_out[class_name] += 1
                        self.stats.class_counts_out[class_name] += 1
                        self.crossed_ids[track_id] = cy
 
    def display_counts(self, frame):
        """
        Display the counts of objects that have crossed the line (both "IN" and "OUT").

        Args:
        frame (ndarray): A single frame from the video to overlay the count text on.
        """
        y_offset = 30
        for class_name, count in self.class_counts_out.items():
            cv2.putText(frame, f"OUT {class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

        y_offset = 30
        for class_name, count in self.class_counts_in.items():
            cv2.putText(frame, f"IN {class_name}: {count}", (self.frame_width - 250, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30

    def process_video(self):
        """
        Processes the video frame by frame, tracks objects, and saves the output video.

        Continuously processes the video, applies object detection and tracking, and saves the processed frames
        into the output video file.
        """
        while self._cap.isOpened():
            
            ret, frame = self._cap.read()
            if not ret:
                break
            
            self.process_frame(frame)
            if self.display==True:
                self.display_counts(frame)

            # Save processed frame
            self.out.write(frame)

        # Release resources after processing
        self._cap.release()
        self.out.release()


    # TODO : fix the multithreading 
    # def speed_process(self):
    #     threads = []
    #     while self._cap.isOpened():
    #         ret, frame = self._cap.read()
    #         if not ret:
    #             break

    #         # create a thread for processing each time
    #         thread = threading.Thread(target=self.process_frame, args=(frame,))
    #         threads.append(thread)
    #         thread.start()

    #         # limit the threads running at once 
    #         if len(threads)>5:
    #             for  t in threads:
    #                 t.join()
    #             threads = []


    #         # display the counts if display is true
    #         if self.display :
    #             self.display_counts(frame)

    #         self.out.write(frame) # save processed frame

    #     # wait for all threads to complete before releasing resources
    #     for t in threads:
    #         t.join()

    #     # release resources after processing
    #     self._cap.release()
    #     self.out.release()


