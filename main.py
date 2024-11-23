import cv2 as cv
import os
from detect import run
from picamera2 import Picamera2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
import time
import pathlib

pathlib.WindowsPath = pathlib.PosixPath

sample_images_path = 'data/images'
weights_path = 'best.pt'

def run_picamera(weights, 
                 imgsz=(640, 640), 
                 conf_thres=0.25, 
                 iou_thres=0.45, 
                 device='', 
                 view_img=True, 
                 save_img=False,
                 frame_stride=30):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize Picamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Allow the camera to warm up
    time.sleep(0.1)

    frame_count = 0

    while True:
        # Capture frame-by-frame
        im0 = picam2.capture_array()

        # Only process every nth frame
        if frame_count % frame_stride == 0:
            # Preprocess the frame
            im = cv.resize(im0, imgsz)  # resize
            im = im.transpose((2, 0, 1))  # Change from HWC to CHW
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                im0_copy = im0.copy()
                annotator = Annotator(im0_copy, line_width=3, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_copy.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Display the resulting frame
                if view_img:
                    cv.imshow('Picamera2', im0_copy)

                # Save results (image with detections)
                if save_img:
                    save_path = str(Path('picamera2_output.jpg'))
                    cv.imwrite(save_path, im0_copy)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release resources and close windows
    cv.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    weights_path = 'best.pt'
    run_picamera(weights_path, view_img=True, save_img=False, frame_stride=30)