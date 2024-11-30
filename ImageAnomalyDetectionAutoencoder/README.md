# Autoencoder for Anomaly Detection in Manufacturing Images

This project focuses on developing an autoencoder for anomaly detection in manufacturing images. The autoencoder is trained to reconstruct normal images, with reconstruction error used as a criterion for anomaly detection.

## Dataset Citation
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
IEEE Conference on Computer Vision and Pattern Recognition, 2019


## Project Summary

- **Goal**: Detect anomalies in drug capsule images by identifying high reconstruction errors from an autoencoder model.

- **Dataset**:
  - 219 training images, all taken from consistent conditions (distance, orientation, etc.).
  - Test dataset includes images with and without anomalies.

## Autoencoder Design

- **Architecture**:
  - **Encoder**:
    - 3 convolutional layers with strided convolutions for downsampling.
    - Channel progression: \(1 \rightarrow 32 \rightarrow 64 \rightarrow 128).
    - ReLU activation and Batch Normalization after each convolution.
    - Dropout applied to encoder to prevent overfitting.
  - **Decoder**:
    - 3 deconvolutional layers mirroring the encoder's channel progression.
    - ReLU activation and Batch Normalization after each layer, except the final layer.
    - Final layer uses a Sigmoid activation to normalize outputs to \([0, 1]\).
  - **Loss Function**: Mean Squared Error (MSE) between input and reconstructed images.

- **Design Decisions**:
  - Initial experiments with max pooling layers were reverted in favor of strided convolutions, which produced sharper reconstructions.
  - Depth of the encoder-decoder architecture was fine-tuned to avoid overfitting, given the small dataset size.
  - Batch size decreased from 32 to 16, improving reconstruction quality for the small dataset.

## Training Process

- **Optimization**:
  - Optimizer: Adam with weight decay (L2 regularization).
  - Learning Rate: \(0.001\) with ReduceLROnPlateau scheduler to dynamically adjust learning rate based on validation loss.
- **Data Augmentation**:
  - Due to the uniformity of the dataset, the only transforms applied were converting to grayscale and resize to 512x512.
- **Epochs**: Trained for 100 epochs, saving the best model based on training loss.

## Anomaly Detection

- **Reconstruction Error**:
  - Reconstruction error is calculated as the pixel-wise mean squared error between the input and reconstructed images.
  - A threshold is used to classify images as normal or anomalous.

- **Threshold Tuning**:
  - Histograms of reconstruction errors guided threshold selection.
  - Experimented with thresholds (\(0.0001\), \(7.5e-05\), \(5e-05\)) to optimize precision, recall, and F1 score.
  - The **optimal threshold** depends on the requirements of the problem:
    - **Minimizing false negatives**: Use a lower threshold to ensure anomalies are detected, at the cost of more false positives.
    - **Minimizing false positives**: Use a higher threshold to avoid false alarms, at the cost of missing some anomalies.

- **Performance Metrics**:
  - Accuracy, Precision, Recall, and F1 Score evaluated on the test set.

## Results

- **Best Reconstruction Quality**:
  - Achieved after reducing batch size to 16.
  - Small reconstruction errors for normal images (\(< 0.00075\)).

- **Anomaly Detection Performance**:
  - Precision, recall, and F1 scores varied significantly with the threshold.
  - Threshold tuning remains critical for optimizing detection performance.

## Future Work

- Explore more advanced architectures like variational autoencoders.
- Leverage transfer learning with pre-trained models to enhance feature extraction, especially given the small dataset size.
- Increase dataset size and diversity to improve model robustness.
- Automate threshold selection using ROC curves or Precision-Recall curves.

## Reconstructed Image
![Reconstructed Image](AutoencoderReconstructedImage.png)

