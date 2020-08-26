# Binocular-Siamese-like-Network-for-RDR-Detection
A siamese-like CNN for diabetic retinopathy detection using binocular fudus images as input.

## Notes
1. The dataset is downloaded from the official website of the Kaggle [Diabetic Retinopathy Detection Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection).
2. The pre-trained weight of the backbone network can be found [here](https://github.com/fchollet/deep-learning-models/releases/tag/v0.2).
3. By running the file "generate_list", you can generate the lists containing index paths and labels of the fundus images, which are used for training and validation. 
4. The true labels of the test set provided by Kaggle are unknown, thus our work focus more on the performance on the validation set.

## Citation
Please cite the following paper if you use the code:

[X. Zeng, H. Chen, Y. Luo and W. Ye, "Automated Diabetic Retinopathy Detection Based on Binocular Siamese-Like Convolutional Neural Network," in IEEE Access, vol. 7, pp. 30744-30753, 2019, doi: 10.1109/ACCESS.2019.2903171.](https://ieeexplore.ieee.org/document/8660434)
