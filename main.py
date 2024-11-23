import cv2 as cv
import numpy as np
import os
from detect import run
from picamera2 import Picamera2, Preview
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
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    # Allow the camera to warm up
    time.sleep(0.1)

    while True:
        # Capture frame-by-frame
        frame = picam2.capture_array()

        # Run the detection
        run(weights=weights_path, source=frame)

        # Display the frame
        cv.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cv.destroyAllWindows()
        
            
if __name__ == '__main__':
    main()