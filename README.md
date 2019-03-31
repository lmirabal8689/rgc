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

## Detecting objects in an image
After the environment is set up, images can be passed for detection.
```
python3 detection.py splash --weights=/path/to/car.h5 --image=<URL or path to file>
```



## Training a model
This is an example of object annotation. Each object is bound and tagged (tagging is helpful for multiple classes within the same dataset).
![](/assets/annotation.jpg)




```
python3 detection.py train --dataset=/path/to/dataset --weights=coco
```