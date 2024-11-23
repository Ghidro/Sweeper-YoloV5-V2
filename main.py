import cv2 as cv
import numpy as np
import os
from detect import run
from picamera2 import Picamera2
import time
import matplotlib.pyplot as plt

sample_images_path = 'data/images'
weights_path = 'best.pt'

camera_port = 0

def main():
    # for image in os.listdir(sample_images_path):
    #     image_path = os.path.join(sample_images_path, image)
    #     run(weights=weights_path, source=image_path)
        
    #     print(f'Processing {image_path}...')
    
    # Initialize the Picamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Allow the camera to warm up
    time.sleep(0.1)

    # Run YOLO inference on the camera feed and save the results
    run(weights=weights_path, source=0, view_img=False, save_img=True, project='runs/detect', name='exp', exist_ok=True)

if __name__ == '__main__':
    main()