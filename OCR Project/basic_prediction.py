"""
OCR(Optical Character Recognition) System - Part which Predict Image

Created by Mehmet Zahid GENÇ - 2022

"""



import torch
import numpy as np
import cv2
from operator import itemgetter

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', device='cpu')

img = cv2.imread('img path')


def get_info(img, model):
    global img2
    """
    @variable results   : The results of analysis of model prediction
    @variable data      : Tabulated version of the results obtained using pandas
    @variable data_list : We convert the data table into a list in order to be able to sort
    @variable average   : The average of the accuracy values of the characters detected as a result of the analysis of the picture
    @variable text      : It is used to make the sequence obtained as a result of placing the characters detected as a result of the
                          analysis into text
    """

    results = model(img, size=700)

    data = results.pandas().xyxy[0]

    """
    We are deleting the columns that we will not use from the data table.
    The information we use is "the class name-[2]", "the confidence value-[1]", and "the x_min value-[0]".
    """

    cv2.namedWindow('IMG', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Img', cv2.WINDOW_NORMAL)


    data.__delitem__('xmax')
    data.__delitem__('ymax')
    data.__delitem__('ymin')
    data.__delitem__('class')

    data_list = data.values.tolist()

    data_list = sorted(data_list, key=itemgetter(0))  # We sort by x_min values

    sum_of_confidence = 0  # Total accuracy value in an image
    text = ''

    for element in data_list:
        """
        element[0] = x_min
        element[1] = confidence value
        element[2] = class name
        """

        sum_of_confidence = sum_of_confidence+element[1]
        text = text+element[2]+'-'

    text = text[:len(text)-1]  # We perform the removal of the trailing "-"

    # We calculate the "average accuracy value"
    if len(data_list) == 0:
        average = 0
    else:
        average = sum_of_confidence / len(data_list)

    print(average, text)

    cv2.imshow('IMG', np.squeeze(results.render()))

    cv2.waitKey(0)

get_info(img, model)