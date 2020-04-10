# TinyImagenet
Classification task on images.

## Models used:
&bull; VGG11 (re-written by me)
<br>&bull; Resnet18
<br>&bull; Resnet34
<br>&bull; Wide Resnet50_2

<br>Overall 7 experiments were made and the best results gives us:

<br><b>Model:</b> Wide Resnet50_2
<br><b>Avg. train accuracy:</b> 89.59.
<br><b>Avg. train loss:</b>     4.1420e-01.
<br><b>Avg. validation accuracy:</b> 68.07.
<br><b>Avg. validation loss:</b> 1.5945e+00.

## Link to data:
<br>After you download, create a directory named ```data``` and unpack downloaded data to the ```/data``` folder. 
<br>Preprocessed data(validation is ready to load using ImageFolder)
<br> https://drive.google.com/open?id=1-uImm2MAFUcnwwW4V5Wmae6xVoiVp_WY
<br>Raw data(vaalidation is not sorted). I have used ```sort_elem_toFolder.py```, which is place by path "/scr/"
<br> https://drive.google.com/open?id=1znM8juVg352A6805NXEUuBZ8MM9T5lq_

## Link to download best model weights:
<br>: https://drive.google.com/open?id=1mCNqoRgKPGdJ3bGcqgjIxVA9KDYJIYml
Create 2 folders in ```/models/```: "load path" and "checkpoints" and after you download weights put it into ```/models/load_path```

<br> Model weights will be provided only for the best model, but if needed you can send e-mail to ```sanjar.dosov97@gmail.com``` and I will send best epoch snapshot for other experiments.

## Augmentation used:
I have used 2 type of augmentation:
<br><b>First:</b> RandomCrop(56), RandomHorizontalFlip(), ToTensor(), Normalize(). The same was used on validation set
<br><b>Second:</b> I have used test time augmentation with TenCrop technique, but it gave worse results.

## Guide:
<br>First install all requirements:
<br>```pip install -r requirements.txt```
<br>To run train script run:
```python /src/main.py```
There you will find all arguments, also two possible training functions with different augmentation
