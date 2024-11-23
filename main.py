import cv2 as cv
import numpy as np
import os
from detect import run

sample_images_path = 'data/images'
weights_path = 'best.pt'

def main():
    for image in os.listdir(sample_images_path):
        results = run(weights=weights_path, source=image)
        
        for result in results:
            # preview the results
            cv.imshow('result', result)
            cv.waitKey(0)
            cv.destroyAllWindows()
            
if __name__ == '__main__':
    main()