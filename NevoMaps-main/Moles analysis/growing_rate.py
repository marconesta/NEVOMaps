from PIL import Image, ImageEnhance, ImageDraw
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, measure, exposure, color

#from skimage.morphology import disk

from sklearn.cluster import KMeans
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
from sticker_identification import Sticker
import pandas as pd
import name
from moles_identification import NevoMaps

PATH = ""
def calculate_growth_rate(old_mask, current_photo, old_pixel_areas, old_mm_areas):
    """
    Assumo che in input all'oggetto della classe venga data:
        una maschera binaria segmentata di una parte del corpo (OLD MASK)
        l'immagine nuova a colori appena scattata
        aree in pixel foto vecchia
        aree in mm foto vecchia
    L'immagine nuova a colori viene analizzata con script principale --> ottengo così info immagine recente
    Queste info le confronto con le info immagine vecchia
    PROBLEMI possibili
        1)nell'immagine appena fatta viene individuato un numero diverso di nei --> il confronto come si fa??
            la cosa buona è che in questo modo potremmo dire che ci sono dei nei nuovi...
        2) questo script si baserà sul fatto che i nei verranno sempre analizzati con lo stesso ordine anche in immagini diverse

    :return:
    """
    image = io.imread(PATH)

    sticker = Sticker(image)

    s = NevoMaps(img=image, sticker_obj=sticker)
    array_cropped_moles, array_segmented_moles, mask_container, moles_pixel_area, mole_mm_area = s.analizza_immagini()

    growing_rate_array = []
    # ho tutte info immagine attuale, quelle vecchie le ho passate come parametro prese dal database
    if len(old_pixel_areas) == len(moles_pixel_area):
        for i, (old_mole_pixel, new_mole_pixel) in enumerate(zip(old_pixel_areas, moles_pixel_area)):
            growing_rate_array.append(0)
            if new_mole_pixel >= (old_mole_pixel + 0.1*old_mole_pixel):
                print(f"Il neo {i+1} ha cambiato dimensione")
                growing_rate = round((new_mole_pixel-old_mole_pixel)/old_mole_pixel*100)
                growing_rate_array.append(growing_rate)
    else:
        print("Ci sono dei nei nuovi")

    return growing_rate_array



