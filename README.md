# AlexNet on CIFAR-10

This repository implements AlexNet, a deep convolutional neural network, trained on the CIFAR-10 dataset using PyTorch. The project replicates key aspects of the original AlexNet paper (Krizhevsky et al., 2012), including data augmentation, SGD optimization, and learning rate scheduling, with adaptations for CIFAR-10’s 32x32 images. The model achieves strong performance.

## Features
- **AlexNet Architecture**: Customized with batch normalization, adjusted padding (2 for Conv1/2), and batch norm before pooling.
- **Data Augmentation**: Random crops (`RandomCrop(32, padding=4)`) and horizontal flips (`RandomHorizontalFlip()`), per the paper.
- **Optimizer**: SGD (`lr=0.001`, `momentum=0.9`, `weight_decay=0.0005`).
- **Scheduler**: `ReduceLROnPlateau` for adaptive learning rate reduction.
- **Validation**: Split from training set to monitor performance and guide scheduling.
- **Visualization**: Matplotlib plots of random test images with actual vs. predicted classes (~4/5 correct).


## Results
- **Training (20 epochs)**:
  - Train Accuracy: 90.14%
- **Testing (10,000 images)**:
  - Test Accuracy: 85.96%
  - Visualization: ~4/5 correct predictions per 5-image batch

## Notes

- The original learning rate used in the paper was `lr = 0.01` but was changed to `lr = 0.001` as the model's training was unstable and lead to poor results both on train and test set.
- The original paper used 1000 classes as opposed to 10 used in this repository.
- The batch-size used here is 64 (in the original paper it was 128).
- The blurry images in visualisation are due to scaling of images from 32*32 to 224 * 224. 

## References
  * [Original Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
  * [PyTorch Documentation](https://docs.pytorch.org/docs/2.6/)
  * [Writing AlexNet from Scratch in PyTorch](https://medium.com/@whyamit404/writing-alexnet-from-scratch-in-pytorch-15dfbf06fefc)
  * [AlexNet Architecture Explained](https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5)
  * [Krish Naik's Video Explaining the Architecture](https://www.youtube.com/watch?v=7LQSdPjWjdA)
  * [Krish Naik's Implementation in Keras](https://github.com/krishnaik06/Advanced-CNN-Architectures/blob/master/Transfer%20Learning%20Alexnet.ipynb)
