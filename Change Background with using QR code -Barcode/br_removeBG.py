import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import br_config

"""
This function takes two input; 
       Image that is we are gonna use as background image 
       Original frame
"""

def bg_remove(img, imgBG):
    segment = SelfiSegmentation()

    imgBG = cv2.resize(imgBG, (br_config.horizontal_axis, br_config.vertical_axis))

    frame_out = segment.removeBG(img, imgBG, threshold=br_config.bg_remove_threshold)

    return frame_out