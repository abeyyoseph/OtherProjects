# Object Detection Model: ISL Hand Gesture Recognition

## Model Overview
This repository contains the training and evaluation of an object detection model using **YOLOv8** for recognizing hand gestures from the Hand Gesture Recognition Computer Vision dataset on Roboflow (https://universe.roboflow.com/lebanese-university-grkoz/hand-gesture-recognition-y5827/?ref=pyimagesearch). The model is trained with the aim of detecting five distinct classes of hand gestures.

## Design Decisions
The following decisions were made during the design and training process:

### Model Architecture
- The nano **YOLOv8** model was chosen due to its fast inference time and suitability for object detection tasks.
- The model was initialized with **pretrained weights** (`yolov8n.pt`) to take advantage of transfer learning, as the dataset is relatively small.

### Hyperparameters
- **Learning rate (`lr0`)**: 0.001 
- **Optimizer**: **AdamW** was used due to its efficient handling of sparse gradients.
- **Batch size**: 32.
- **Image size**: 416x416.
- **Weight decay**: 0.001.
- **Augmentation**: Mosaic augmentation, flipping, scaling, and translation was used.
- **Learning Rate (`lr0`)**: The learning rate was increased from 0.001 to 0.01 to speed up convergence.
- **Cosine Annealing**: The **cosine annealing learning rate schedule** was enabled with `cos_lr`, allowing the learning rate to decay smoothly during training.
- **Learning Rate Range Factor (`lrf`)**: Set to 0.01 to control the adjustment of the learning rate during training.


### Training Setup
- **Epochs**: 25 epochs.
- **Warm-up**: A 3-epoch warm-up phase for better convergence.
- **Freeze layers**: 10 initial layers were frozen to preserve the pretrained weights during the early stages of training.

---

## Validation Results

### Overall Results
- **Inference Speed**: 24.8ms
- **Precision**: 0.782
- **Recall**: 0.738
- **mAP @ IoU=0.50**: 0.818
- **mAP @ IoU=0.50:0.95**: 0.68


#### Per-Class Results:
| Class   | mAP @ IoU=0.50 | mAP @ IoU=0.50:0.95 |
|---------|--------------------|--------------------|
| Class "one" | 0.927 | 0.721 |
| Class "two" | 0.779 | 0.671 |
| Class "three" | 0.827 | 0.683 |
| Class "four" | 0.63 | 0.544 |
| Class "five" | 0.926 | 0.779 |

---

## Future Work
- Investigate **hyperparameter tuning** for optimal performance, including learning rate adjustments, weight decay, and optimizer settings.
- Explore **different models** (e.g., YOLOv10/11) to see if they outperform YOLOv8 for this dataset.


