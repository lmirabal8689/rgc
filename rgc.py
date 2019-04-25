#!/usr/bin/env python3

# ICT-435 Senior project
# Written by Laurence Mirabal and Jonahlyn Gilstrap
# To run:
# python rgc.py -h for help
# python rgy.py -i <input file name or url>

import logging
logging.getLogger('tensorflow').disabled = True

import os
from pathlib import Path
import cv2
import numpy as np
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
import MySQLdb as mariadb
import skimage.draw
import sys
import getopt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

VWIDTH = 800
VHEIGHT = 450


class MaskRCNNConfig(mrcnn.config.Config):
  """ Mask-RCNN Configuration"""

  NAME = "coco_pretrained_model_config"
  IMAGES_PER_GPU = 1
  GPU_COUNT = 1
  # COCO dataset has 80 classes + one background class.
  NUM_CLASSES = 1 + 1
  # The detection confidence can be adjusted for easier detection.
  DETECTION_MIN_CONFIDENCE = 0.6


def insert_data(output_data):
  # Connect to the mariadb.
  mariadb_connection = mariadb.connect(host=config.get('mysqlDB', 'host'),
                                      database=config.get('mysqlDB', 'database'),
                                      user=config.get('mysqlDB', 'user'),
                                      password=config.get('mysqlDB', 'password'))
  cursor = mariadb_connection.cursor()

  try:
    add_record = ("INSERT INTO data "
                      "(date, vehicles, filename) "
                      "VALUES (%s, %s, %s)")
    cursor.execute(add_record, output_data)
    mariadb_connection.commit()
      
  except mariadb.Error as err:
    print(err)
  except:
    print("Unknown error occured")
  finally:
      if(mariadb_connection is not None):
          print("*Successfully inserted data!")
          print("Closing database connection.")
          print()
          mariadb_connection.close()


def get_car_boxes(boxes, class_ids, scores):

  car_boxes = []
  car_scores = []
  for i, box in enumerate(boxes):
      # Use the vehicle class.
      if class_ids[i] in [1]:
          car_boxes.append(box)
          car_scores.append(scores[i])

  return np.array(car_boxes), np.array(car_scores)


def start_detection(input_image, output_directory):
  
  # Download the pretrained weights file for car.h5.
  CURRENT_DIR = Path(".")
  MODEL_DIR = os.path.join(CURRENT_DIR, "logs")
  MODEL_PATH = os.path.join(CURRENT_DIR, "car.h5")

  # If model not found then download coco model.
  if not os.path.exists(MODEL_PATH):
    mrcnn.utils.download_trained_weights(MODEL_PATH)

  # Create the model to be used for object detection
  model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
  model.load_weights(MODEL_PATH, by_name=True)

  print()
  print("*Opening image.")
  image_copy = skimage.io.imread(input_image)
  cv2.imwrite("temp.png", image_copy)
  image_cropped = cv2.imread("temp.png")

  # Hard coded list of the points that cover the road
  pts = np.array([[145,250],[350,450],[750,410],[230,245]])
  pts2 = np.array([[145,250],[350,450],[750,410],[230,245]])

  print("*Cropping image area of interest.")
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  croped = image_cropped[y:y+h, x:x+w].copy()

  # Mask the outer cropped area.
  pts = pts - pts.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

  # Copy the mask to a new image in memory.
  new_image_cropped = cv2.bitwise_and(croped, croped, mask=mask)

  # Detect vehicles on cropped image.
  print("*Detecting vehicles.")
  results = model.detect([new_image_cropped], verbose=0)
  result = results[0]
  car_boxes, car_scores = get_car_boxes(result['rois'], result['class_ids'], result['scores'])
  print("%d vehicles detected on road." % (len(car_boxes)))

  # Draw the bounding boxes onto the original image.
  cv2.polylines(image_cropped, [pts2], True, (0,0,255),2)

  # Offset included from the cropped image dimensions
  font = cv2.FONT_HERSHEY_SIMPLEX
  for box, score in zip(car_boxes, car_scores):
    y1, x1, y2, x2 = box
    cv2.rectangle(image_cropped, (x1+145, y1+240), (x2+145, y2+240), (0,255,0), 2)
    cv2.rectangle(image_cropped, (x1+145,y1+240), (x1+200,y1+225), (255,0,0), -1)
    cv2.putText(image_cropped,str(score)[:5],(x1+145,y1+240), font, .6, (255,255,255), 1, cv2.LINE_AA)


  #for score in car_scores:

  # Save file to disk
  output_img = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR)
  
  # Generate the file name based on its name. (Should be a datetime.extension)
  file_name = "output_" + input_image[-20:-4] + ".png"

  # Parse the file name (datetime) into a usable date tim.
  image_time_stamp = input_image[-20:-4]
  image_time_stamp = image_time_stamp[:-3] + ':' + image_time_stamp[14:]
  image_time_stamp = image_time_stamp[:-6] + ' ' + image_time_stamp[11:]
  image_time_stamp = image_time_stamp + ':00'

  # Write the image to file in the output directory /var/www/images/
  print("*Writing image to file.")
  cv2.imwrite(file_name, output_img)

  insert_data((image_time_stamp, len(car_boxes), file_name))


def main(argv):
  # Initialize the i/o variables 
  input_image = None
  output_directory = None

  try:
    opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print ("Argument not recognized, try -h for help")
    sys.exit(2)
  # Parse arguments and set variables.
  for opt, arg in opts:
    if opt == '-h':
      print()
      print("-o 'image output directory' (exclude for default directory var/www/image)")
      print("-i 'image file directory or url'")
      print()
      sys.exit()
    elif opt in ("-i", "--ifile"):
      input_image = arg
    elif opt in ("-o", "--ofile"):
      output_directory = arg

  # Check arguments
  if input_image is None:
    print("Please include the -i argument, enter -h for help.")
    sys.exit()
  
  if output_directory is None:
    output_directory = "/var/www/images/"

  print ('Input file -> ', input_image)
  print ('Output directory -> ', output_directory)

  # If arguments are valid, pass to begin detection.
  start_detection(input_image, output_directory)


if __name__ == "__main__":
  main(sys.argv[1:])


#create table data ( `id` int(11) NOT NULL AUTO_INCREMENT, `date` datetime NOT NULL, `vehicles` smallint NOT NULL, `filename` varchar(512) NOT NULL, PRIMARY KEY (`id`));