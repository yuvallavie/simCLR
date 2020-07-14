# simCLR
simCLR linear evaluation using pre-trained weights by https://github.com/google-research/simclr
- ResNet50x1 https://drive.google.com/file/d/13x2-QBIF1s6EkTWf1AjHEGUc4v047QVF/view?usp=sharing
  Please download this file and extract its contents to the "checkpoints" folder.

this file is added as a resource to my seminar presentation at Bar-Ilan University.
Author : Yuval Lavie

File locations:
<br>
    - ../data
    - ../models
    - ../checkpoints

Were using a pre-trained ResNet50 (x1) as the encoding-head and finetuning a multiclass logistic regressor over it.
the dataset is STL-10.

It's obvious that supervised representation learning is still better but unsupervised representation is catching up and may soon replace the need for many labeled samples.

