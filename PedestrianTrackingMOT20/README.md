# Object Tracking with YOLO and ByteTrack

## Overview
This project focuses on using YOLOv8 for object detection and ByteTrack for multi-object tracking on the MOT20 dataset. The goal was to develop a tracker
capable of detecting and tracking persons in a video feed, achieving high precision while balancing recall and Multi-Object Tracking Accuracy (MOTA) metrics.

## Design Decisions
1. **Model Selection:**
   - **YOLOv8** was chosen as the object detection model due to its speed and accuracy for real-time applications.
   - **ByteTrack** was selected as the tracker because of its efficiency and ability to handle occlusions and multi-object scenarios.

2. **Data Preprocessing:**
   - The **MOT20 dataset** was used, with a focus on tracking people in the provided test sequences.
   - The training dataset was processed to ensure all detections belonged to the same class, as only a single class was tracked (person).
   - Due to the size of the datasets and limited compute resources, I split the MOT20-02 training set into a "training" and 
   "validation" dataset to train the model. I then used the smaller MOT20-01 dataset as my test set for evaluating the object 
   detection model and then the tracking model.
   
3. **Object Detection Model Training Hyperparameters:**
   - **Epochs:** 10
   - **Batch Size:** 16
   - **Image Size (imgsz):** 1280
   - **Device:** CPU
   - **Frozen Layers:** 10 (freeze 10 layers in the YOLO model for transfer learning)
   - **Optimizer:** AdamW
   - **Learning Rate (lr0):** 0.001
   - **Weight Decay:** 0.001
   - **Data Augmentation:** Enabled
   - **Confidence Threshold (conf):** 0.3 (Min confidence threshold for detections)
   - **NMS (Non-Maximum Suppression):** Enabled (to suppress redundant overlapping boxes)

4. **Tracking Setup:**
   - **ByteTrack Tracker:** Fine-tuned with parameters like `track_thresh` for tracking thresholds, `min_hits`, and `max_age` for handling occlusions and missed detections.

## Results
### YOLOv8 Model Evaluation on Test Set:
- **Precision:** 0.959
- **Recall:** 0.281
- **mAP50:** 0.623
- **mAP50-95:** 0.49

### ByteTrack Evaluation on MOT20-01:
- **MOTA:** 20.8%
- **IDF1:** 34.6%
- **Precision:** 100.0%
- **Recall:** 20.9%

### Notes:
- Precision is high, indicating that the model is good at detecting persons when it does detect them, but recall is low, 
suggesting that many objects are not detected.
- Tracking performance (MOTA) is decent, but there is room for improvement in the recall and IDF1 scores.

## Future Work
1. **Model Improvement:**
   - **Fine-tune YOLOv8** for more epochs or on a more diverse dataset to improve recall.
   - Use more of the MOT20 training sets to train the model.
   - Further tune learning rate and confidence threshold during training.
   
2. **ByteTrack Adjustments:**
   - Further adjust ByteTrack parameters such as `track_thresh`, `max_age`, and `min_hits` to optimize tracking performance, especially in crowded or occluded scenarios.
   
3. **Data Augmentation:**
   - Implement additional augmentation techniques to improve detection in varied conditions.

