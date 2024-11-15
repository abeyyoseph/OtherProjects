# Binary Classification  Task

## Prompt
You are provided with a dataset of human joint annatotions. Each annotation includes a label indicating whether the human is doing lawn work or playing a sport. __Your task is to build a classifier that predicts whether a human is doing lawn work or playing a sport based on their joint annotations.__  Your classifier should perform as best as possible on new data, evaluated by F1 score

In your on-site interview, you will be asked to explain your model and your process for obtaining it. You will also be asked how you expect your model to perform on new data and why. Limit your work on this task to no more than 2 hours, honor system

## About the dataset
The dataset is stored in `mpii_dataset.csv`. Each row corresponds to a single person. Each row includes 15 joint annotations:
- head
- upper_neck
- thorax
- left/right shoulder (l/rsho)
- left/right elbow (l/relb)
- left/right wrist (l/rwri)
- left/right hip (l/rhip)
- left/right knee (l/rknee)
- left/right ankle (l/rank)

The below image shows the location each joint:
![093672999](https://github.com/LeonardGrazianGC/cv-take-home-task/assets/127145473/a7bd7894-9d57-41a4-a9dc-7bd77f1ef7b5)

Each annotation includes 3 fields:
- x: the horizontal distance for the __left side__ of the image, measaured in pixels
- y: the vertical distance from the __top__ of the image, measured in pixels
- vis: integer flag indicating if the joint is visible. `1` for visible, `0` for not visible

Columns of the dataset follow this naming convention: `<joint name abbreviation>_<field>`. For example, `lank_x`, `rhip_vis`, `upper_neck_y`. The dataset contains 46 columns, 3 for each joint and 1 for the target label

The dataset includes a label indicating if the person is doing lawnwork or playing a sport. This label is stored as an integer flag in the `sport` column. It is `1` if the person is playing a sport and `0` if the person is doing lawnwork 

The dataset includes 694 examples of lawnwork and 1,587 examples of sports

## Python Example
In addition to the dataset, this repo includes a simple python example in `example.py`. It loads the dataset, trains a simple classifier on the entire dataset, and prints out the model's 
confusion matrix and F1 score on the entire dataset. The libaries required to run the example are located in `example_requirements.txt`

## Bibliography
Thanks to the creators of the MPII Human Post Dataset
```
@inproceedings{andriluka14cvpr,
    author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
    title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June}
}
```

And thanks to these authors for enhancing the readability of the dataset
```
@inproceedings{Shukla_2022_BMVC,
    author    = {Megh Shukla and Roshan Roy and Pankaj Singh and Shuaib Ahmed and Alexandre Alahi},
    title     = {VL4Pose: Active Learning Through Out-Of-Distribution Detection For Pose Estimation},
    booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
    publisher = {{BMVA} Press},
    year      = {2022},
    url       = {https://bmvc2022.mpi-inf.mpg.de/0610.pdf}
}

@inproceedings{9706805,
    author={Shukla, Megh},
    booktitle={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
    title={Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?}, 
    year={2022},
    volume={},
    number={},
    pages={2021-2030},
    doi={10.1109/WACV51458.2022.00208}
}

@inproceedings{9523037,
    author={Shukla, Megh and Ahmed, Shuaib},
    booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
    title={A Mathematical Analysis of Learning Loss for Active Learning in Regression}, 
    year={2021},
    volume={},
    number={},
    pages={3315-3323},
    doi={10.1109/CVPRW53098.2021.00370}
}
```
