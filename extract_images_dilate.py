import cv2
import os
from pathlib import Path
import copy

import argparse
from glob import glob

COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

OUTPUT_DIR = 'images'
STEPS_DIR = 'steps'

def morphShape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

def extractImagesFromFile(filename):
    
    trace_width = 2

    # Load image
    img = cv2.imread(filename)
    img_orig = copy.copy(img)

    # Grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Binary image
    thr, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Dilated image
    dilatation_size = 3
    dilation_shape = morphShape(2)
    kernel = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=2)

    # Find contours, obtain bounding box, extract and save region of interest (ROI)
    roi_number = 1
    cnts = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c) # Coordinates, width, height of ROI
        area = cv2.contourArea(c) # Area of ROI found

        if area > 3000 and w < 560:
            cv2.rectangle(img, (x, y), (x+w, y+h), COLOR_BLUE, trace_width)
            ROI = img_orig[y:y+h, x:x+w]
            cv2.putText(img=img, text=f"a={int(area):.2f}, w={int(w):.2f}", org=(x, y-5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=COLOR_BLUE, thickness=2)

            # Write ROI
            path_roi = os.path.join(OUTPUT_DIR, f"{Path(filename).stem}_{roi_number}.jpg")
            cv2.imwrite(path_roi, ROI)
            roi_number += 1

    # Write steps
    gray_3_channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(img=gray_3_channel, text=f"threshold:{thr}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=COLOR_BLUE, thickness=2)
    thresh_3_channel = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
    dilate_3_channel = cv2.cvtColor(img_dilate, cv2.COLOR_GRAY2BGR)
    im_h = cv2.hconcat([gray_3_channel, thresh_3_channel, dilate_3_channel, img])
    path_steps = os.path.join(STEPS_DIR, f"{Path(filename).stem}_steps.jpg")
    cv2.imwrite(path_steps, im_h)


def main(files):
    # Create the output directory if it does not exist
    output_path = Path.cwd() / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    steps_path = Path.cwd() / STEPS_DIR
    steps_path.mkdir(exist_ok=True)

    for file in files:
        print(f"Processing {file}")
        if Path(file).is_file():
            extractImagesFromFile(file)
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
