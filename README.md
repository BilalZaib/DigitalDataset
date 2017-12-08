# DigitalDataset
Digital data-set generated using Python PIL and OS fonts for AI semester project.

## Folder Structure
 **test-data-519/:** This folder contains all the dataset images in their respective folders.
 
 **TESTSet.py:** This code is used to generate the dataset. This takes all fonts from the operating system and writes them to the 10x10 image to generate the dataset image.
 
 **MLP.py:** This code contains the code to load the dataset (in Python) and then test that dataset in Keras and Scikit-learn.

## Result
This result were collected using [MLP.py](MLP.py), goal of this result were to match Scikit-learn and Keras result under same architecture. While with default parameter and same architecture, result difference is high (Scikit-learn is better).

### Using Scikit-learn
```
Number of samples in training set: 4150, number of samples in test set: 1040
('Train Accuracy:', 0.98168674698795177)
('Test Accuracy:', 0.85673076923076918)
('Layers:', 3)
('Output Layer size: ', 10)
('Number of Iteration: ', 200)
('Output Activation: ', 'softmax')
```

### Using Keras 
```
Number of samples in training set: 4150, number of samples in test set: 1040
('Train accuracy:', 0.98168674713157744)
('Test accuracy:', 0.85961538461538467)
('Layers:', 3)
('Output Layer size: ', 10)
('Number of Epochs: ', 200)
('Output Activation: ', 'softmax')
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
