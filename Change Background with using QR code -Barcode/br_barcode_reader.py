import cv2
import numpy as np
from pyzbar.pyzbar import decode

"""
This function read barcode and return data. Also it surrounds the barcode with line, put data into image  
"""

def barcode(img):
    for barcode in decode(img):
        my_data = barcode.data.decode('utf-8')

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)

        pts2 = barcode.rect
        cv2.putText(img, my_data, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 255), 2)

        return my_data