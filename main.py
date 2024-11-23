import cv2 as cv
import numpy as np
import os
from detect import run
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

sample_images_path = 'data/images'
weights_path = 'best.pt'

camera_port = 0

def main():
    # for image in os.listdir(sample_images_path):
    #     image_path = os.path.join(sample_images_path, image)
    #     run(weights=weights_path, source=image_path)
        
    #     print(f'Processing {image_path}...')
    
    # Initialize the camera
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # Allow the camera to warm up
    time.sleep(0.1)

    # Capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        # Run the detection
        run(weights=weights_path, source=image)

        # Display the frame
        cv.imshow("Frame", image)
        
        # Clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cv.destroyAllWindows()
        
            
if __name__ == '__main__':
    main()