import cv2

import os
from pathlib import Path
import copy

import argparse
from glob import glob

COLOR_GREEN = (36, 255, 12)
COLOR_BLUE = (255, 0, 0)

OUTPUT_DIR = 'images'
TRACING_DIR = 'tracing'


def extractImagesFromFile(filename, tracing=True, extend=True):
    
    trace_width = 2
    offset = 10

    # Load image
    img = cv2.imread(filename)
    height, width = img.shape[:2]

    img_orig = copy.copy(img)

    # Grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Blurred image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) 

    # Binary image
    img_thresh = cv2.threshold(img_blur, 230, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find contours, obtain bounding box, extract and save region of interest (ROI)
    img_number = 1
    cnts = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c) # Coordinates, width, height of ROI
        area = cv2.contourArea(c) # Area of ROI found

        if area > 2000:
            # Basic contours
            cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_GREEN, trace_width)

            # Extended contours
            if extend: 
                y1 = y-offset
                y2 = y+h+offset
                x1 = x-offset
                x2 = x+w+offset
                if y1 < 0:
                    y1 = y
                if y2 > height:
                    y2 = y+h
                if x1 < 0:
                    x1 = x
                if x2 > width:
                    x2 = x+w
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BLUE, trace_width)
                ROI = img_orig[y1:y2, x1:x2]
            else:
                ROI = img_orig[y:y+h, x:x+w]
            
            # Write ROI
            path_roi = os.path.join(OUTPUT_DIR, f"{Path(filename).stem}_{img_number}.jpg")
            cv2.imwrite(path_roi, ROI)
            img_number += 1

    # Write tracing
    if tracing:
        path_roi = os.path.join(TRACING_DIR, Path(filename).stem + '_trace.jpg')
        cv2.imwrite(path_roi, img)


def main(files):
    tracing_enabled = True
    extend = True

    # Create the output directory if it does not exist
    output_path = Path.cwd() / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)


    if tracing_enabled:
        tracing_path = Path.cwd() / TRACING_DIR
        tracing_path.mkdir(exist_ok=True)

    for file in files:
        print(f"Processing {file}")
        if Path(file).is_file():
            extractImagesFromFile(file, tracing_enabled, extend)
        else:
            print(f"Invalid file: {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("fileNames", nargs='*') 
    args = parser.parse_args()  
    
    fileNames = list()  
    for arg in args.fileNames:  
        fileNames += glob(arg)  
    main(fileNames)
