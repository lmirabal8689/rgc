# Traffic Detection
[Project link](http://rgc.nmsu.edu/)
![](/assets/dashboard.png)
[Dashboard webapp by Jonahlyn](https://github.com/jonahlyn/dashboard)

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

In addition to these requirements, you will need the pretrained model that detects vehicles for this specific purpose. It can be downloaded [here](https://laurencemirabal.com:4444/index.php/s/sB9iCe9S53APyH5). The model was created from pre-trained coco weights. It trained for 15 epochs with about 50 test images and 30 validation images each containing < 10 objects.

You can find out more about coco [here](http://cocodataset.org/#home).


## Example Output
![](/assets/output_example_15_epoch_60.png)

## Detecting objects in an image
After the environment is set up, images can be passed for detection.
```
python3 rgc.py -i <image url or directory>
```



## How it works
First an image is read and then a hard coded list of points is used to capture the road and crop that selection to a new image. This isolates the area we are interested in and ignores the parked cars. The area within the blue polygon is the area we want to detect vehicles. This also gives us a smaller image to work with. The detection runs and gets a list of bounding boxes for the roi's which is then used to draw onto the original image. The image is saved onto the /var/www/images/ directory and an entry is created in the database to reference the file and total vehicles along with a timestamp.

![](/assets/cropped_image.png)
![](/assets/final_image.png)
## Training a model
This is an example of object annotation. Each object is bound and tagged (tagging is helpful for multiple classes within the same dataset). The image only shows a few vehicles annotated, this is because the set of images I used have many instances where vehicles did not move. My technique involved only annotating new vehicles per image to avoid duplicate objects. I have not data to suggest this was a good idea but it seems to have worked fine. [This](http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html) webpage is what I used to annotate each image.
![](/assets/annotation.PNG)


To train the model you will need to specify the dataset directory. Additionally, it must contain both a "val" and "train" directory with your images and the JSON file containing the annotation data. To train, I suggest using a GPU because it is an extremely intensive task. Further instruction is required for using a GPU and is not covered in this document nor are examples of training. See the Matterport repository for examnples.

This project is based on matterport Mask_RCNN.
[Mask_RCNN](https://github.com/matterport/Mask_RCNN)