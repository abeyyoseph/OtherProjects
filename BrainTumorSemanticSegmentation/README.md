# Brain Tumor Segmentation using FCN-ResNet50

## Task Description
The objective of this project is to perform semantic segmentation on brain CT scans (from https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation/data) to identify and segment tumor regions. The dataset consists of grayscale images with corresponding masks, where pixel values indicate the presence of a tumor.

## Approach
### Model
- Transfer learning is performing using the ResNet50 model, a convolutional network pretrained on ImageNet, as the backbone.
- The final layers were adapted for binary segmentation (tumor vs. background).

### Data Preprocessing
- Images are resized to `256x256`.
- Data augmentation techniques applied:
  - Horizontal flip
  - Small rotation
  - Translation
  - Color jitter
- Images are normalized using ImageNet mean and standard deviation.

### Training Details
- **Loss Function:** Dice Loss 
- **Optimizer:** AdamW
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Weight Decay:** 0.001
- **Number of Epochs:** 10

## Results
### Training, Validation, and Test
- Ten training epochs were executed and the validation loss was stored for each epoch. The best performing model on the validation set was saved for further use on the test set.
- The final test loss was 0.4169.

## Observations & Next Steps
- Validation loss improved over the training epochs, with the best model achieving `0.4026`.
- Further tuning of hyperparameters, such as learning rate schedules, dropout, or alternative architectures, may further improve performance.
- Due to compute resources, only ten training epochs were performed. With more resources/time, the model would have been trained for a longer amount of time (20-30 epochs).

- An example of the model's predicted mask can be seen below:
![Prediction](Prediction.jpg)
