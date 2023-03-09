##Imports
import os
import csv
import pandas as pd
import random as rnd
from PIL import Image
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import Dataset


##Convert image size
#Original 1003 x 124

# rename files with the following (different in bash)
# arguments <arg> are to be replaced (and remove <>)
# power shell: $get-childitem *.<ext> | foreach { rename-item $_ $_.Name.Replace("<string_from>", "<string_to>") }

# To create multiple random data
def random_crop(path):
    vertical_offset = rnd.randint(0,60)
    horizontal_offset = rnd.randint(0,939)
    im = Image.open(path)
    im_cropped = im.crop((horizontal_offset + 0, vertical_offset + 0, horizontal_offset + 64, vertical_offset + 64))
    return im_cropped

# Prepares the dataset
# To run the custom_gan, images have to be in a sub-directory of the dataset folder
def prepareRandomDataset(input_path, output_path, reuse=5): # reuse is the number of the we crop a same image
    for file in os.listdir(input_path):
        path = os.path.join(input_path, file)
        if os.path.isfile(path):
            for i in range(0,reuse):
                im = random_crop(path)
                save_path = os.path.join(output_path, '', str(i) + '_' + file)
                im.save(save_path)


##Create label file

def createLabelsCSV(dataset_path, output_path):
    csv_file = open(output_path + r'\labels_nonseg.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_file)
    for file in os.listdir(dataset_path):
        writer.writerow([file, '1'])
    csv_file.close()