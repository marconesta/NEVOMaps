from skimage import io, measure
from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as nd
import pandas as pd
from skimage.morphology import binary_closing, binary_opening, disk
import name

class Sticker():

    def __init__(self, img):
        self.img = img
        self.area_sticker_pixel, self.sticker_mask = self.identify_sticker()

    def identify_sticker(self):
        # Definisci la gamma di colori verde da filtrare
        lower_green = np.array([40, 50, 50])  # Valori HSV minimi per il verde
        upper_green = np.array([80, 255, 255])  # Valori HSV massimi per il verde

        # Converti l'immagine in formato HSV
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Applica la maschera per filtrare gli oggetti verdi
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        # Applica un'operazione morfologica per migliorare la maschera
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        for _ in range(3):
            closed_mask = nd.binary_closing(mask, np.ones((7, 7)))

        # count numbers of 1
        area_all_stickers = np.count_nonzero(closed_mask == 1)

        area_sticker = int(area_all_stickers/name.NUM_STCIKER)


        return area_sticker, closed_mask