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
- **Augmentation**: Mosaic augmentation was used, set to a factor of `1.0` to improve model generalization.

### Data Augmentation
- **Mosaic**: This technique was enabled to randomly sample and combine multiple images, improving the model's robustness and diversity during training.
- **Flip, Scale, and Translate**: Basic augmentations to simulate different orientations and scales of objects.
  
### Training Setup
- **Epochs**: 25 epochs.
- **Warm-up**: A 3-epoch warm-up phase for better convergence.
- **Freeze layers**: 10 initial layers were frozen to preserve the pretrained weights during the early stages of training.

### Evaluation Metrics
- **Average Precision (AP)** at IoU=0.50, 0.75, and 0.50:0.95.
- **Average Recall (AR)** at various IoU thresholds.
  
---

## Training Results

### First Set of Results
- **Inference Speed**: 19.2ms
- **Average Precision (AP) @ IoU=0.50:0.95**: 0.598
- **Average Precision (AP) @ IoU=0.50**: 0.770
- **Average Precision (AP) @ IoU=0.75**: 0.748
- **Average Recall (AR) @ IoU=0.50:0.95 (all areas, maxDets=100)**: 0.638
- **Average Recall (AR) @ IoU=0.50:0.95 (large areas, maxDets=100)**: 0.641

#### Per-Class Results:
| Class   | AP @ IoU=0.50:0.95 | AR @ IoU=0.50:0.95 |
|---------|--------------------|--------------------|
| Class "five" | 0.6476 | 0.7152 |
| Class "four" | 0.5613 | 0.6000 |
| Class "one" | 0.6042 | 0.6429 |
| Class "three" | 0.5468 | 0.5750 |
| Class "two" | 0.6290 | 0.6571 |

### Changes Between First and Second Set of Results
The following modifications were made to the model between the first and second set of results:

1. **Learning Rate (`lr0`)**: The learning rate was increased from 0.001 to 0.01 to speed up convergence.
2. **Cosine Annealing**: The **cosine annealing learning rate schedule** was enabled with `cos_lr`, allowing the learning rate to decay smoothly during training.
3. **Learning Rate Range Factor (`lrf`)**: Set to 0.01 to control the adjustment of the learning rate during training.

### Second Set of Results
- **Inference Speed**: 20.7ms
- **Average Precision (AP) @ IoU=0.50:0.95**: 0.504
- **Average Precision (AP) @ IoU=0.50**: 0.653
- **Average Precision (AP) @ IoU=0.75**: 0.632
- **Average Recall (AR) @ IoU=0.50:0.95 (all areas, maxDets=100)**: 0.571
- **Average Recall (AR) @ IoU=0.50:0.95 (large areas, maxDets=100)**: 0.574

#### Per-Class Results:
| Class   | AP @ IoU=0.50:0.95 | AR @ IoU=0.50:0.95 |
|---------|--------------------|--------------------|
| Class "five" | 0.5935 | 0.7087 |
| Class "four" | 0.3290 | 0.3545 |
| Class "one" | 0.5811 | 0.6857 |
| Class "three" | 0.3634 | 0.4125 |
| Class "two" | 0.6533 | 0.6929 |

---

## Summary and Comparison
- **Inference Speed**: The second set of results has a slight increase in inference time (20.7ms vs. 19.2ms).
- **Overall Performance**: The first set generally shows better performance across most metrics, particularly in terms of **Average Precision (AP)** and **Average Recall (AR)** at various IoU thresholds.
  - The **first set** outperforms the **second set** in **AP @ IoU=0.50:0.95** (0.598 vs. 0.504), **AP @ IoU=0.50** (0.770 vs. 0.653), and **AP @ IoU=0.75** (0.748 vs. 0.632).
  - The **second set** shows some improvement for **Class "two"**, but overall, the first set is the more balanced and higher-performing model.

---

## Future Work
- Investigate **hyperparameter tuning** for optimal performance, including learning rate adjustments, weight decay, and optimizer settings.
- Explore **different models** (e.g., YOLOv10/11) to see if they outperform YOLOv8 for this dataset.


