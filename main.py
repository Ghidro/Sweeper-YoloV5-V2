import cv2 as cv
import numpy as np
import os
from detect import run

sample_images_path = 'data/images'
weights_path = 'best.pt'

def main():
    for image in os.listdir(sample_images_path):
        image_path = os.path.join(sample_images_path, image)
        run(weights=weights_path, source=image_path)
        
        print(f'Processing {image_path}...')
        
            
if __name__ == '__main__':
    main()