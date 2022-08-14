""""
OCR(Optical Character Recognition) System - Part which Analyzing and Deleting Image

Created by Mehmet Zahid GENÇ - 2022


^ Preliminary Information ^
___________________________

* An object of the Predictor class can be created by taking many parameters.
* After the object is created, a "self-method" is created by using the methods in it.
* This method consists of "analysis of images", "order of accuracy" and "deletion of unnecessary images".



^ Information of This Script's Method ^
_______________________________________

@Method: __init__                   : Introducing the necessary parameters. It has default values

@Method: get_model                  : Loading and returning the torch model using the parameters defined in the initializer method

@Method: get_info                   : It takes the image as input and the model which returned from the get_model method.
                                      In this method, the image is analyzed and then the average accuracy value and object rank are returned as output.

@Method: get_path                   : Returns a path that is updated every day using the actual_path retrieved
                                      by the __init__ method and using the day, month, year information

@Method: get_last_3_images_path     : Using the path returned from the get_path method,
                                      the paths of the last 3 images in the file are added to a list and the list is returned.

@Method: delete_unnecessary_images  : Using the get_info method, the images assigned to the list are analyzed and the necessary data is obtained.
                                      This data is stored in a list and it is arranged in a row to decide which images should be deleted,
                                      then the images that are not deemed necessary (2 images with the lowest accuracy value among the 3 images)

@Method: run_prediction_function    : It is a complex method. Although it is the method of the Predictor class,
                                      it is a combination of the other methods of this class.


"""



"""

^ Information about Libraries which is used in OCR System Project ^
___________________________________________________________________

@Lib datetime-date       : It is used to obtain day month year information for this project.

@Lib os                  : It is used for operations such as using path to access files for this project.

@Lib torch               : It is used to load the model to be used in image analysis for this project.

@Lib cv2                 : It is used to load the image to be analyzed

@Lib operator-itemgetter : It is used while we put the characters detected in the picture in order and 
                           perform the deletion process by ordering accuracy.
                           
"""

from datetime import date
import os
import torch
import cv2
from operator import itemgetter
import numpy as np


class Predictor:
    def __init__(self, actual_path="OCR", model_type="yolov5", dataset_type="custom",
                 pt_file_path="yolov5//yolov5s.pt", source="local", device="cpu"):

        """

        @attribute actual_path      :  Path information of the main folder where the images are saved
        @attribute model_type       :  The type of model we need to specify when loading the model with the torch library
        @attribute dataset_type     :  The type of dataset we need to specify when loading the model with the torch library
        @attribute pt_file_path     :  The path of the model we produced ourselves
        @attribute source           :  Information of source
        @attribute device           :  The parameter that specifies which device (cpu - gpu) we use the model we use while analyzing the pictures.

        """

        self.__actual_path = actual_path
        self.__model_type = model_type
        self.__dataset_type = dataset_type
        self.__pt_file_path = pt_file_path
        self.__source = source
        self.__device = device

    def get_model(self):

        """

        @variable  model : Model loaded using attributes in the __init__ method ("model type", "dataset type", "pt file path", "source", "device")

        """

        model = torch.hub.load(self.__model_type, self.__dataset_type, path=self.__pt_file_path, source=self.__source,
                               device=self.__device)

        return model # Return model to use in run_prediction_function method


    def get_info(self, img, model):

        """

        @variable results   : The results of analysis of model prediction

        @variable data      : Tabulated version of the results obtained using pandas

        @variable data_list : We convert the data table into a list in order to be able to sort

        @variable average   : The average of the accuracy values of the characters detected as a result of the analysis of the picture

        @variable text      : It is used to make the sequence obtained as a result of placing the characters detected as a result of the
                              analysis into text


        """

        results = model(img, size=640)

        data = results.pandas().xyxy[0]

        """
        We are deleting the columns that we will not use from the data table.
        The information we use is "the class name-[2]", "the confidence value-[1]", and "the x_min value-[0]".
        """

        data.__delitem__('xmax')
        data.__delitem__('ymax')
        data.__delitem__('ymin')
        data.__delitem__('class')

        data_list = data.values.tolist()

        data_list = sorted(data_list, key=itemgetter(0)) # We sort by x_min values

        sum_of_confidence = 0 # Total accuracy value in an image
        text = ''

        for element in data_list:
            """
            element[0] = x_min
            element[1] = confidence value
            element[2] = class name
            """

            sum_of_confidence = sum_of_confidence+element[1]
            text = text+element[2]+'-'

        text = text[:len(text)-1] # We perform the removal of the trailing "-"
        
        cv2.imshow('IMG', np.squeeze(results.render()))

        cv2.waitKey(0)

        # We calculate the "average accuracy value"
        if len(data_list) == 0:
            average = 0
        else:
            average = sum_of_confidence / len(data_list)

        return average, text  # Average is the average of confidence value of detections, text is the queue of the objects in the image



    def get_path(self):
        """
        @variable year   : year info of today
        @variable month  : month info of today
        @variable day    : day info of today
        @variable path   : specific folder path created using information obtained ("year", "month", "day")
        """
        today_date = date.today()

        year = today_date.year
        month = today_date.month
        day = today_date.day

        path = self.__actual_path+f'//{year}//{month}//{day}'

        return path


    def get_last_3_images_path(self, path):

        """
        All files in the specific file for that day be accessed and
        each time the names of the last 3 images save to the file_list and then return file_list
        """

        file_list = []

        for filename in os.listdir(path):
            file_list.append(filename)

        file_list = file_list[-3:]

        return file_list


    def delete_unnecessary_images(self, confidence_removing_list):
        """
        information saved into confidence_removing_list like as [avg, img_path, text2]
        sorted with using 0. index (average confidence value) because we want to delete 2 image with worst accuracy
        """

        confidence_removing_list = sorted(confidence_removing_list, key=itemgetter(0))
        
        print(f'After sorted (low to high accuracy): {confidence_removing_list}')

        for liste in confidence_removing_list[:2]:
            path = liste[1]
            os.remove(path)
            
        return confidence_removing_list

    def run_prediction_function(self):
        """
        This method is our main method.

        It is a set of methods created using other methods of the class

        After all the operations are completed,
        the sequence information of the characters detected in the picture with the highest accuracy result among the 3 pictures is returned as output.

        """
        path = self.get_path()
        file_list = self.get_last_3_images_path(path)

        confidence_removing_list = []  # We create a list to store confidence values and img_path that is will used to delete unnecessary images

        model = self.get_model()

        for file_name in file_list:
            img_path = path+f'//{file_name}'
            img = cv2.imread(img_path)
            avg, text2 = self.get_info(img=img, model=model)

            instantaneous_values = [avg, img_path, text2]

            confidence_removing_list.append(instantaneous_values)

        confidence_removing_list = self.delete_unnecessary_images(confidence_removing_list=confidence_removing_list)

        result_of_analysis = confidence_removing_list[2][2]

        return result_of_analysis


"""
We could use the information we want instead of using the default information when creating the object.
"""

predictor = Predictor() # Create a Predictor object.

result = predictor.run_prediction_function() # Result of analyzing of method of predictor object

print(result) # Just display the result in terminal
