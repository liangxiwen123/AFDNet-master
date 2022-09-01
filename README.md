# AFDNet
### Representitive Results
![representive_results](/AFDNet-master/AFDNet-master/assets/1.png)

![representive_results](/AFDNet-master/AFDNet-master/assets/2.png)

![representive_results](/AFDNet-master/AFDNet-master/assets/3.png)
### Overal Architecture
![architecture](/AFDNet-master/AFDNet-master/assets/4.png)

![architecture](/AFDNet-master/AFDNet-master/assets/5.png)

![architecture](/AFDNet-master/AFDNet-master/assets/6.png)
## Environment Preparing
```
python3.5
```
You should prepare at least 3 1080ti gpus or change the batch size. 


```pip install -r requirement.txt``` </br>
```mkdir model``` </br>
Download VGG pretrained model from [[Google Drive 1]](https://drive.google.com/file/d/1IfCeihmPqGWJ0KHmH-mTMi_pn3z3Zo-P/view?usp=sharing), and then put it into the directory `model`.

### Training process
Before starting training process, you should launch the `visdom.server` for visualizing.

```nohup python -m visdom.server -port=8097```

then run the following command

```python scripts/script.py --train```

### Testing process


Create directories `../test_dataset/testA` and `../test_dataset/testB`. Put your test images on `../test_dataset/testA` (And you should keep whatever one image in `../test_dataset/testB` to make sure program can start.)

Run

```python scripts/script.py --predict```

### Dataset preparing

Training data First Download training data set from https://daooshee.github.io/BMVC2018website/. Save training pairs of our LOL dataset ( including 500 paired images) LOL Testing data also come from there. Put low-light images in `../final_dataset/testA` and high-light images in `../final_dataset/testB`.

Testing data [[Google Drive]](https://drive.google.com/open?id=1PrvL8jShZ7zj2IC3fVdDxBY1oJR72iDf) (including LIME, MEF, NPE, VV, DICP)

And [[BaiduYun]](https://github.com/TAMU-VITA/EnlightenGAN/issues/28) is available now thanks to @YHLelaine!


### Reference

Code borrows heavily from https://github.com/VITA-Group/EnlightenGAN.
