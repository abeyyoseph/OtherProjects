import numpy as np
import os
from time import time
import cv2
import torch
import ultralytics
from ultralytics import YOLO
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, yolo_model_name, capture_index):

        print("Initializing tracker object")
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model(yolo_model_name)
        
        self.CLASS_NAMES_DICT = self.model.model.names
        
        self.box_annotator = sv.BoxAnnotator(sv.Color.GREEN, thickness=2, text_thickness=1, text_scale=0.5)

        self.object_tracker = DeepSort(max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None)
        self.thr = 0.3 # detection threshold
        
    def load_model(self, yolo_model_name):
        print("Loading yolo model:", yolo_model_name)
        model = YOLO(yolo_model_name)  
        model.fuse()

        return model
    
    def predict(self, frame):
    
        results = self.model(frame)
        
        return results
    
    def generate_bboxes(self, results):
        '''
        transforms coordinates from YOLO detector format: [x1, y1, x2, y2]
        to DeepSORT Tracker format: [left, top, w, h]
        returns a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
        '''

        cords = results[0].boxes.xyxy.tolist()
        conf = results[0].boxes.conf.tolist()
        classes = results[0].boxes.cls.tolist()

        detections = []

        for i in range(len(results[0])):
            if conf[i] > self.thr:
                #transforming coordinates
                cords[i][2] = int(cords[i][2] - cords[i][0]) #width = x2 - x1
                cords[i][3] = int(cords[i][3] - cords[i][1]) #height = y2 - y1
                #generating detections list
                detections.append((cords[i], conf[i], classes[i]))

        return detections
    
    def plot_tracks(self, tracks, frame):
        
        tracks = [track for track in tracks if track.is_confirmed()]

        if np.any(tracks): #checks if array has confirmed trakcs to plot, if not, returns original frame as is

            cords = np.array([track.to_ltrb() for track in tracks])
            ids = np.array([track.track_id for track in tracks])
            # Extract & Setup detections for visualization
            tracks_sv = sv.Detections(
                        xyxy=cords,
                        tracker_id=ids,
                        )
            
            # Format labels
            # self.labels = ["ID:" + str(tracker_id) for _,_,_,_,tracker_id in tracks_sv]
            self.labels = ["ID:" + str(tracker_id[-2]) for tracker_id in tracks_sv]
            
            # Annotate and display frame
            frame = self.box_annotator.annotate(scene=frame, detections=tracks_sv, labels=self.labels)
        
        return frame
    
    def __call__(self):

        if self.capture_index:
            #Initialize the input and output videos
            video_path = os.path.join('.', 'data', 'video', 'cars.mp4')
            video_out_path = os.path.join('.', 'data', 'video', 'out1.mp4')
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (width, height))
     

        #Iterate through the video frame by frame   
        while True:
            
            start_time = time()
            
            #Read in a frame
            ret, frame = cap.read()

            #Break the loop if there are no more frames to consume
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            results = self.predict(frame)
            detections = self.generate_bboxes(results)
            
            tracks = self.object_tracker.update_tracks(detections, frame=frame)
            #plot tracks func, takes frame and tracks and returns an annotated frame like:
            frame = self.plot_tracks(tracks, frame)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
                
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)

            if self.capture_index:
                cap_out.write(frame) #write to file

            if cv2.waitKey(5) & 0xFF == 27: #27 -> escape key                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":    

    ultralytics.checks()
    tracker = Tracker("yolov8n.pt", capture_index=1)
    tracker()