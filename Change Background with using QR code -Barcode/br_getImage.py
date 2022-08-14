import urllib.request
import br_config

"""
From br_barcode_reader, barcode function return a url information in br_main. This function takes this url and install the image.
We define file_path and filename to determine where img is gonna be install. 
"""

def getIMage(image_url):
    full_path = br_config.file_path + br_config.filename + br_config.img_type
    urllib.request.urlretrieve(image_url, full_path)