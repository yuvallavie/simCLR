# simCLR
simCLR linear evaluation using pre-trained weights by https://github.com/google-research/simclr
- ResNet50x1 https://drive.google.com/file/d/13x2-QBIF1s6EkTWf1AjHEGUc4v047QVF/view?usp=sharing
  Please download this file and extract its contents to the "checkpoints" folder.

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
Just place the extracted ResNet50x1.pth file in the checkpoints folder and run the file
<br>
python main.py

<br>
Requirements:
<br>
- PyTorch
It's obvious that supervised representation learning is still better but unsupervised representation is catching up and may soon replace the need for many labeled samples.

