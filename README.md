<h1>Retinal Optical Coherence Tomography Classification</h1>
<div align='center'>
    <img src = "data_train/NORMAL/NORMAL-1384-1.jpeg" width=300px>
    <img src = "data_train/CNV/CNV-13823-1.jpeg" width=300px>
</div>

<div>
    In this project, I utilized the ResNet101 architecture to peform image classification on the <a href="https://www.kaggle.com/datasets/paultimothymooney/kermany2018">Retinal OCT dataset</a> into four different classes: normal, choroidal neovascularization (CNV), diabetic macular edema (DME), and Drusen eye disease.
</div></br>

<p>
    The ultimate goal of this project is not to create a state-of-the-art classification pipeline, but rather a personal exploration into model compression (specifically: quantization-aware training, and iterative pruning) for the purpose of deploying the model to a Raspberry Pi 5 for inference. Thus, I opted to choose the ResNet101 architecture over more-modern architectures such as MobileNet or EfficientNet due to its simplicity and good documentation. Initial training on this dataset yielded ~95% accuracy on the validation set as shown in figure 1.
</p>
<div align="center">
    <img src="figures/normal_metrics.png" width=500px></br></br>
</div>
<p>
    Initial training on this dataset yielded ~95% accuracy on the validation set. To prepare the model for deployment in embedded environments, it is a necessity to convert from 32-bit floating bit numbers to an 8-bit integer. However, directly making this conversion after training induces quantization noise, which results in a degradation of classification accuracy. 
</p>

<p>
    In order to make the model more-robust to quantization noise, I simulated quantization during model training using torchao. The resulting peformance can be seen in figure 1.
</p>
