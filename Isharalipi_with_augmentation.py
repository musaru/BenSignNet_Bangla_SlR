import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

import cv2
import numpy as np


class skinDetector(object):

    # class constructor
    def __init__(self, imageName):
        self.image = cv2.imread(imageName)
        if self.image is None:
            print("IMAGE NOT FOUND")
            exit(1)
        # self.image = cv2.resize(self.image,(600,600),cv2.INTER_AREA)
        self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        self.binary_mask_image = self.HSV_image

    # ================================================================================================================================
    # function to process the image and segment the skin using the HSV and YCbCr colorspaces, followed by the Watershed algorithm
    def find_skin(self):
        self.__color_segmentation()
        output=self.__region_based_segmentation()
        return output

    # ================================================================================================================================
    # Apply a threshold to an HSV and YCbCr images, the used values were based on current research papers along with some
    # empirical tests and visual evaluation
    def __color_segmentation(self):
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(self.HSV_image, lower_HSV_values, upper_HSV_values)

        self.binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        #self.binary_mask_image = mask_HSV

    # ================================================================================================================================
    # Function that applies Watershed and morphological operations on the thresholded image
    def __region_based_segmentation(self):
        # morphological operations
        image_foreground = cv2.erode(self.binary_mask_image, None, iterations=3)  # remove noise
        dilated_binary_image = cv2.dilate(self.binary_mask_image, None,
                                          iterations=3)  # The background region is reduced a little because of the dilate operation
        ret, image_background = cv2.threshold(dilated_binary_image, 1, 128,
                                              cv2.THRESH_BINARY)  # set all background regions to 128

        image_marker = cv2.add(image_foreground,
                               image_background)  # add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
        image_marker32 = np.int32(image_marker)  # convert to 32SC1 format

        cv2.watershed(self.image, image_marker32)
        m = cv2.convertScaleAbs(image_marker32)  # convert back to uint8

        # bitwise of the mask with the input image
        ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        output = cv2.bitwise_and(self.image, self.image, mask=image_mask)

        # show the images
        #self.show_image(self.image)
        #self.show_image(image_mask)
        #return image_mask
        #self.show_image(output)
        return output

    # ================================================================================================================================
    def show_image(self, image):
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
#https://github.com/Jeanvit/PySkinDetection/blob/master/src/jeanCV.py
#path='data'
#path='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/ASL/Alphabets_Colored'
#path= 'F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Jpsword'
#path='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/All_Data'
#path2='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/Segmented'
#path2='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/mask_YCBRHSV'
#path2='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/mask_ycbr'
#path2='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/mask_HSV'

path='F:/PhD/Professor Work/Sign/MyPhDProject/Dataset/Bangla/Ishara_Lipi/Alphabet/'

#
Img_size = 60

target = []
dgjwImages = []
flat_data = []

training_data = []
label = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        # print(dirname)
        for (direcpath, direcnames, files) in os.walk(path + "/" + dirname):
            for file in files:
                actual_path = path + "/" + dirname + "/" + file
                # label=label_img(dirname)
                path1 = path + "/" + dirname + '/' + file
                # img=cv2.imread(path1)

                detector = skinDetector(path1)
                img = detector.find_skin()

                # path3 = path2 + "/" + dirname+"/"+file
                # print(path3)
                # cv2.imwrite(path3, img)

                img_resized = cv2.resize(img, (Img_size, Img_size))

                flat_data.append(img_resized.flatten())
                target.append(label)

                img_resized = expand_dims(img_resized, 0)
                img_resized = expand_dims(img_resized, 3)
                # print(img_resized.shape)
                img_resized=img_resized.reshape(1,Img_size,Img_size,3)

                datagenrot = ImageDataGenerator(rotation_range=30)
                igr1 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(rotation_range=45)
                igr2 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(rotation_range=60)
                igr3 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(rotation_range=90)
                igr4 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot= ImageDataGenerator(zoom_range=[0.5, 1.0])
                igz = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(horizontal_flip=True)
                igf1 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(vertical_flip=True)
                igf2 = datagenrot.flow(img_resized, batch_size=1)

                #datagenrot = ImageDataGenerator(width_shift_range=[-200, 200])
                #isft1 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot= ImageDataGenerator(height_shift_range=0.5)
                isft2 = datagenrot.flow(img_resized, batch_size=1)

                datagenrot = ImageDataGenerator(rescale=1./255)
                ibr = datagenrot.flow(img_resized, batch_size=1)

                for i in range(9):
                    batch1r1 = igr1.next()
                    Images = batch1r1[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size, 3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    batch1r2 = igr2.next()
                    Images = batch1r2[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size, 3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    batch1r3 = igr3.next()
                    Images = batch1r3[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size, 3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    batch1r4 = igr4.next()
                    Images = batch1r4[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size,3)
                    flat_data.append(Images.flatten())
                    target.append(label)
                    # plt.imshow(Images)
                    # plt.show()

                    batch2 = igz.next()
                    Images = batch2[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size, 3)
                    flat_data.append(Images.flatten())
                    target.append(label)
                    # plt.imshow(Images)
                    # plt.show()

                    batch3f1 = igf1.next()
                    Images = batch3f1[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size,3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    batch3f2 = igf2.next()
                    Images = batch3f2[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size,3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    # batch4s1 = isft1.next()
                    # Images = batch4s1[0].astype('uint8')
                    # Images = Images.reshape(90, 90, 3)
                    # flat_data.append(Images.flatten())
                    # target.append(label)


                    batch4s2 = isft2.next()
                    Images = batch4s2[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size,3)
                    flat_data.append(Images.flatten())
                    target.append(label)

                    batch5 = ibr.next()
                    Images = batch5[0].astype('uint8')
                    Images = Images.reshape(Img_size, Img_size,3)
                    flat_data.append(Images.flatten())
                    target.append(label)

        label = label + 1

Ishara_lipiSA_data = np.array(flat_data)
Ishara_lipiSA_target = np.array(target)
print(Ishara_lipiSA_data.shape)
#Bangla_limages = np.array(images)
#
import pickle
pickle_out=open("Ishara_lipiSA_data.pickle", "wb")
pickle.dump(Ishara_lipiSA_data ,pickle_out)
pickle_out.close()
#
#
pickle_out=open("Ishara_lipiSA_target.pickle", "wb")
pickle.dump(Ishara_lipiSA_target,pickle_out)
pickle_out.close()