# InceptionNet implementation in PyTorch


This is an implementation of InceptionNet architecture proposed by Christian Szegedy et al. in the paper [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842v1.pdf) using PyTorch. The files contain implementation of GoogLeNet, which is based on the Inception V1 module, I will later add Inception V2 and V3 modules as well.

The Jupyter Notebook contains details about the architecture and implementation steps, the Python script contains the code.

The Jupyter Notebook and Python files also contain image pipeline for the Tiny ImageNet dataset, however I did not train the model on the dataset due to hardware limitations. If you wish to train the model using the Tiny ImageNet dataset then you should download it from [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), I did not include the dataset in the repository as it is quite large, however it is very straight forward to download and train the model after you download it, just mention the file path of the `tiny-imagenet-200` folder in the `DATA_PATH` in `main.py`.


The original paper used Polyak averaging to create the model at inference time, however I omitted that in my implementation, as it would be computationally more expensive to keep track of all weights and I am limited by hardware.


**NOTE:** This repository contains the model architecture of the original paper as proposed by Szegedy et al., the original architecture was trained on the ILSVRC 2014 dataset consisting of 1.2 million images distributed among 1000 classes. While the InceptionNet architecture is a good model for image classification tasks, the hyperparameters such as number of activation maps, kernel size, stride, etc must be tuned according to the problem.

<div>
<img src="https://cdn.discordapp.com/attachments/418819379174572043/1082981014516805712/googlenet.jpg" width="1100" alt = "Going Deeper with Convolutions, Christian Szegedy et al.">
</div>
