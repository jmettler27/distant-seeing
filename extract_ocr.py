import cv2
import copy
import numpy as np
import os
import argparse

import pytesseract
from pytesseract import Output

from pathlib import Path
from glob import glob

COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

OUTPUT_DIR = 'images'
STEPS_DIR = 'steps'


def has_alnum(string):
    """ """
    for c in string:
        if c.isalnum():
            return True
    return False


def find_text_in(img_bin):
    """ """
    img_with_boxes = cv2.cvtColor(img_bin, cv2.COLOR_BGR2RGB)
    
    data = pytesseract.image_to_data(image=img_bin, lang='fra', output_type=Output.DICT)
    text_boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h, c, text) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i], data['conf'][i], data['text'][i])
        if (
            text.strip()
            and (int(c) > 0 and has_alnum(text))
        ):
            text_boxes.append((x, y, w, h))
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color=COLOR_GREEN, thickness=1)
    return img_with_boxes, text_boxes


def text_mask(img_bin, text_boxes):
    """ """
    mask = np.full(img_bin.shape[:2], 255, dtype="uint8")
    for (x, y, w, h) in text_boxes:
        mask[y:y+h, x:x+w] = 0
    return mask


def dilate_image(img, dilation_size, dilation_shape, iterations):
    """ """
    kernel = cv2.getStructuringElement(dilation_shape, (2*dilation_size+1, 2*dilation_size+1), (dilation_size, dilation_size))
    return cv2.dilate(img, kernel, iterations)


def concat_images(*argv):
    """ """
    images = list(argv)
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
    return cv2.hconcat(images)


def extract_images_from(filename):
    # Load image
    img = cv2.imread(filename)
    img_orig = copy.copy(img)

    # Grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Binary image
    thr, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.putText(img=img_gray, text=f"threshold: {thr}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=COLOR_BLUE, thickness=2)
    print(thr)

    # Image with text boxes
    img_with_boxes, text_boxes = find_text_in(img_thresh)

    # Mask image
    img_mask = text_mask(img_thresh, text_boxes)

    # Apply the text mask
    img_thresh_inv = cv2.bitwise_not(img_thresh)
    img_no_text = cv2.bitwise_and(img_thresh_inv, img_mask)

    # Dilated image
    img_dilate = dilate_image(img=img_no_text, dilation_size=9, dilation_shape=cv2.MORPH_RECT, iterations=1)

    # Find contours, obtain bounding box, extract and save region of interest (ROI)
    roi_number = 1
    cnts = cv2.findContours(image=img_dilate, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c) # Coordinates, width, height of ROI
        area = cv2.contourArea(c) # Area of ROI found

        if area > 10000:
            ROI = img_orig[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), color=COLOR_BLUE, thickness=2)
            cv2.putText(img=img, text=f"a={int(area):.2f}", org=(x, y-5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=COLOR_BLUE, thickness=2)

            # Write ROI
            path_roi = os.path.join(OUTPUT_DIR, f"{Path(filename).stem}_{roi_number}.jpg")
            cv2.imwrite(path_roi, ROI)
            roi_number += 1

    # Write steps
    im_h = concat_images(img_with_boxes, img_no_text, img_dilate, img)
    path_steps = os.path.join(STEPS_DIR, f"{Path(filename).stem}_steps.jpg")
    cv2.imwrite(path_steps, im_h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("filenames", nargs='*') 
    args = parser.parse_args()  
    
    filenames = list()  
    for arg in args.filenames:  
        filenames += glob(arg)  
    
    # Create the output directory if it does not exist
    output_path = Path.cwd() / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    steps_path = Path.cwd() / STEPS_DIR
    steps_path.mkdir(exist_ok=True)

    for filename in filenames:
        print(f"Processing {filename}")
        if Path(filename).is_file():
            extract_images_from(filename)
        else:
            print(f"Invalid file: {filename}")
