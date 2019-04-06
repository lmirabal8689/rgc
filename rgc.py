#!/usr/bin/env python3

import logging
logging.getLogger('tensorflow').disabled = True

import os
from pathlib import Path
import subprocess as sp
import datetime as dt
import time as tm
import cv2
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
import MySQLdb as mariadb
from mysql.connector import errorcode
import skimage.draw
import requests
from lxml import html

mariadb_connection = mariadb.connect(user='laurence', password='password', host='192.168.1.90', database='appdb')
cursor = mariadb_connection.cursor()

VWIDTH = 800
VHEIGHT = 450


class MaskRCNNConfig(mrcnn.config.Config):
    """ Mask-RCNN Configuration"""

    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.9


def get_car_boxes(boxes, class_ids):
  '''
  Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
  '''

  car_boxes = []
  for i, box in enumerate(boxes):
      # If the detected object isn't a car/bus/truck, skip it
      if class_ids[i] in [1, 2]:
          car_boxes.append(box)

  return np.array(car_boxes)

# Download the pretrained weights
CURRENT_DIR = Path(".")
MODEL_DIR = os.path.join(CURRENT_DIR, "logs")
MODEL_PATH = os.path.join(CURRENT_DIR, "car.h5")

if not os.path.exists(MODEL_PATH):
  mrcnn.utils.download_trained_weights(MODEL_PATH)

# Create the model to be used for object detection
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(MODEL_PATH, by_name=True)

url2 = 'https://live1.brownrice.com/cam-images/timelapse/ruidosowebcorp2/week-2019-12/'

page = requests.get(url2)
webpage = html.fromstring(page.content)
links = webpage.xpath('.//a/@href')

for x in range(400,450):
  if links[x][-4:] == '.jpg':
    image_link = url2+links[x]
    print(image_link)
    image_copy = skimage.io.imread(image_link)

    cv2.imwrite("temp.png", image_copy)
    image_cropped = cv2.imread("temp.png")

    pts = np.array([[170,270],[350,450],[785,415],[275,260]])
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = image_cropped[y:y+h, x:x+w].copy()
    ## (2) make mask
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    ## (3) do bit-op
    new_image_cropped = cv2.bitwise_and(croped, croped, mask=mask)

    # Detect vehicles and draw bounding boxes
    results = model.detect([new_image_cropped], verbose=0)
    result = results[0]
    car_boxes = get_car_boxes(result['rois'], result['class_ids'])

    for box in car_boxes:
      y1, x1, y2, x2 = box
      cv2.rectangle(image_cropped, (x1+170, y1+260), (x2+170, y2+260), (0,255,0), 2)

    # Save file to disk
    d = dt.datetime.fromtimestamp(tm.time())
    output_img = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
    
    file_name = "output_{}.png".format(d.strftime('%Y%m%d%H%M%S'))
    cv2.imwrite(file_name, output_img)

    output_data = (d, len(car_boxes), file_name)

#    try:
#      add_record = ("INSERT INTO data "
#                        "(date, vehicles, filename) "
#                        "VALUES (%s, %s, %s)")
#      cursor.execute(add_record, output_data)
#      mariadb_connection.commit()

#    except mysql.connector.Error as err:
#        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
#            print("User name or password was incorrect\n")
#        elif err.errno == errorcode.ER_BAD_DB_ERROR:
#            print("Database does not exist\n")
#        else:
#            print(err)
    #finally:
    #    print("Closing database connection")
    #    if(mariadb_connection is not None):
    #        mariadb_connection.close()
        




mariadb_connection.close()