# simCLR

# How to run it
<br>
1. Download the model weights from the link below and extract its content into the checkpoints folder
- ResNet50x1 https://drive.google.com/file/d/13x2-QBIF1s6EkTWf1AjHEGUc4v047QVF/view?usp=sharing
2. Run the system with the either commands 
<br>
For the default run with 20 epochs and a validation size of 0.2:
<br>
- python main.py
For a custom run:
<br>
- python main.py {number of epochs} {size of validation size}



# Description
simCLR linear evaluation using pre-trained weights by https://github.com/google-research/simclr
this file is added as a resource to my seminar presentation at Bar-Ilan University.
Author : Yuval Lavie

File locations:
<br>
    - ../data
<br>    
    - ../models
<br>    
    - ../checkpoints

Were using a pre-trained ResNet50 (x1) as the encoding-head and finetuning a multiclass logistic regressor over it.
<br>
We use the STL-10 dataset, everything other than the model checkpoint will be downloaded automatically.


<br>
Requirements:
<br>
- PyTorch
<br>

It's obvious that supervised representation learning is still better but unsupervised representation is catching up and may soon replace the need for many labeled samples.

