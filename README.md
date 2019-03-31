# Traffic Detection

## Getting Started

Requirements:
- Operating system: Ubuntu 18.04
- User account with sudo privileges
- Python 3.6
- Pip
- Git

Login with a user with sudo privileges and execute the following to install python dependencies (if you have python 3.6 skip the first step and clone the repository):

```
sudo apt-get update && sudo apt-get -y install python3.6 python3-pip
git clone https://github.com/lmirabal8689/rgc.git
```

Mask_RCNN has a whole list of dependencies that can be found in requirements.txt
```
sudo apt-get install -r requirements.txt
``` 

In addition to these requirements, you will need the pretrained model that detects vehicles for this specific purpose. It can be downloaded [here](https://laurencemirabal.com:4444/index.php/s/nptHYAEskHmEK3z). The model was created from pre-trained coco weights. It trained for 15 epocs with 50 test images and 30 validation images each containing > 10 objects.

You can find out more about coco [here](http://cocodataset.org/#home).

## Detecting objects in an image
After the environment is set up, images can be passed for detection.
```
python3 detection.py splash --weights=/path/to/<model_name>.h5 --image=<URL or path to file>
```


## Training a model
This is an example of object annotation. Each object is bound and tagged (tagging is helpful for multiple classes within the same dataset). [This](http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html) is what I used to annotate each image.
![](/assets/annotation.JPG)


To train the model you will need to specify the dataset directory. Additionally, it must contain both a "val" and "train" directory with your images and the JSON file containing the annotation data.
```
python3 detection.py train --dataset=/path/to/dataset --weights=coco
```


This project is based on matterport Mask_RCNN.
[Mask_RCNN](https://github.com/matterport/Mask_RCNN)