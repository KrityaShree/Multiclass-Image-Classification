# Multi-class Image Classification Using Deep Convolutional Neural Networks

## Objective
The objective of this project was to develop an image classification system using deep convolutional neural networks. Initially, the plan was to leverage transfer learning; however, due to resource constraints, a custom model was built instead. The aim was to compare the performance of the custom model with popular pre-trained models and explore object detection alongside multi-class image classification.

## Data Selection
After considering various datasets including DARPA, MNIST Fashion, COCO, and CIFAR100, the team finalized the Tiny ImageNet subset of the ImageNet dataset due to its suitability and manageable size.

## Model Selection
The project explored multiple models including a custom base model, VGG-16, ResNet50, Xception, and EfficientNetV2B3. Each model was evaluated based on its architecture, performance, and computational efficiency.

## Analysis Strategy
The team experimented with different model architectures, hyperparameters, and optimization techniques to improve accuracy. Challenges such as overfitting, resource limitations, and runtime were addressed through techniques like early stopping, hyperparameter tuning, and GPU utilization.

## Individual Contributions
Team members focused on various aspects of the project including model implementation, object detection, pre-trained model integration, and experimentation. Each member contributed to the project's development and problem-solving efforts.

## Final Results
Despite encountering challenges such as resource constraints and overfitting, the team achieved promising results. The custom base model attained a validation accuracy of 57.15%, while pre-trained models like VGG-16 outperformed it with a validation accuracy of 80.08%.

### Base Model:
  - 10 weighted layers
  - Validation Accuracy - 57.15%
  - Traning Accuracy - 64.46%
  - Test Accuracy - 13%
  - Number of epochs - 100  
  - Batch size - 32
  - Number of Images per Batch - 3126

### Pre-trained Models:
- VGG-16 
  - Accuracy : 80.08%
- EfficientNetV2B3 
  - Accuracy : 15.85%
- Xception
  - Accuracy : 63.36%
- ResNet50
  - Accuracy : 42.16%
- EfficientNetB7
  - Accuracy : 0.5%

## Conclusion
The project demonstrated the feasibility of implementing image classification systems using both custom and pre-trained models. It highlighted the importance of model selection, hyperparameter tuning, and resource optimization in achieving optimal performance.

## Future Directions
Future work could involve further experimentation with model architectures, data augmentation techniques, and optimization strategies to enhance accuracy and efficiency. Additionally, exploring larger datasets and advanced object detection methods could lead to more robust and versatile systems.

## Tools Used
1. **Python:** The primary programming language for implementing the deep learning models, data preprocessing, and analysis.
2. **TensorFlow and Keras:** Used for building, training, and evaluating deep neural network models, including both custom and pre-trained architectures.
3. **PyTorch:** Explored for data preprocessing tasks and handling large datasets, providing alternative functionalities to TensorFlow.
4. **ijson Library:** Employed for parsing and processing large JSON files, facilitating data exploration and understanding.
5. **GitHub:** Leveraged for version control, collaboration among team members, and sharing project code and documentation.
6. **Google Colab:** Utilized for running code on GPU-accelerated environments, enabling faster model training and experimentation.
7. **Anaconda:** Used for managing Python environments and installing necessary packages and dependencies.
