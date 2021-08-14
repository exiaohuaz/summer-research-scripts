#!/usr/bin/env python3
#
#  Copyright (c) 2018-2020 Carnegie Mellon University
#  All rights reserved.
#
# Based on work by Junjue Wang.
#
# python3 removedup1.py  -l 2 -p '/Users/Jack/Desktop/captured_processing/Raw/1.2rods' -o 'cropped/No_dup/1.2rods' -t 10 -x 0
#
"""Remove similar frames based on a perceptual hash metric
"""

SRC_DIR = '/home/roger/sterling_fine_grained/combined'


import os
import glob
import argparse
import shutil
import imagehash
from PIL import Image
import numpy as np


DIFF_THRESHOLD = 10


def checkDiff(image_hash, base_image_hash, threshold):
    if base_image_hash is None:
        return True
    if image_hash - base_image_hash >= threshold:
        return True

    return False


def checkDiffComplete(image_hash, base_image_list, threshold):
    if len(base_image_list) <= 0:
        return True
    for i in base_image_list:
        if not checkDiff(image_hash, i, threshold):
            return False
    return True


def main():
    base_image_list = []
    dup_count = 0

    for label in os.listdir(SRC_DIR):
        img_dir = os.path.join(SRC_DIR, label)
        print(label, len(os.listdir(img_dir)))
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            im = Image.open(img_path)
            a = np.asarray(im)
            im = Image.fromarray(a)
            image_hash = imagehash.phash(im)
            if checkDiffComplete(image_hash, base_image_list, DIFF_THRESHOLD):
                base_image_list.append(image_hash)
                # new_name = 'jun21_{}'.format(img_name)
                # shutil.move(img_path, os.path.join(img_dir, new_name))
            else:
                dup_count += 1
                os.unlink(img_path)

    print("Total Dup: ", dup_count)


if __name__ == "__main__":
    main()