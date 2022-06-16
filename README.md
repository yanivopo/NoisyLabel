Noisy Labels - partial-labeling
-------------------- 
### Task definition ### 
In this task CIFAR-10 images are classified based on partial labeling.
`train` - 5K sample, each sample compose of two images from the 
CIFAR10 with one-hot vector contains the two true classes(partial-labeling) 
`validation` - 10K test images if CIFAR-10
  
### Contain ### 
* cifar_cnn.py - The main script for the model.
* utils.py - Utils function. 
* config.py - File configuration. 
* noisy_label.pdf - pdf file with explanation. 
* README.md  

### How to use? ### 
* Download and install Anaconda
* Extract the zip file
* conda create -n noisy_label_env python=3.6
* conda activate noisy_label_env
* conda install tensorflow-gpu
* pip install matplotlib==3.2.1

### Configuration parameter: ### 
* The parameters to config found in the script config.py


The script cifar_cnn contain three model ready to use. 

* my_model - my solution to the problem. 
* upper_bound - The model train with the correct label - upper bound solution
* baseline - A naive baseline to the problem with random choice between the labels of each pair 

### To run example: 
* my_model: python cifar_cnn -m my_model 
* upper_bound: python cifar_cnn -m upper_bound
* baseline: python cifar_cnn -m baseline




