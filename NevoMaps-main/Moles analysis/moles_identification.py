
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
from growing_rate import calculate_growth_rate


class NevoMaps(Sticker):

    def __init__(self, img, sticker_obj, crop_size=200, old_mask = None):
        super().__init__(img)
        self.crop_size = crop_size
        self.image = img
        self.sticker = sticker_obj
        self.old_mask = old_mask

    def analizza_immagini(self):

        # Analizza le immagini utilizzando l'algoritmo K-means per la segmentazione
        array_cropped_moles, array_segmented_moles, binary_full_mask, area_moles_pixel = self.moles_identification()

        # converti pixel in mm^2
        predicted_areas = self.convert_pixel2mm(area_moles_pixel)

        # Prepara i risultati per l'applicazione Flutter
        risultati_flutter = self.prepara_risultati_flutter(binary_full_mask)

        # Restituisci i risultati all'applicazione Flutter
        #self.manda_risultati_a_flutter(risultati_flutter)
        return array_cropped_moles, array_segmented_moles, binary_full_mask, area_moles_pixel, predicted_areas


    def moles_identification(self):
        """
        Funzione che prende immagine e identifica i nei. Croppa sui nei in base al blob detection,
        segmenta usando un K-means, ritorna 3 cose
        # TODO quando croppo immagine potrei salvare dei falsi positivi, con degli algoritmi la situazione è migliorabile
        :param image:
        :return: array contenenti immagini dei nei croppati,
        array binario dei nei croppati, maschera finale binaria
        """

        image_height, image_width, _ = self.image.shape

        detector = self.blob_detector()

        keypoints = detector.detect(self.image)


        # counting number of blobs
        print("Number of blobs detected:", len(keypoints))

       # img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                           #cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # K-
        # Crop the image around each blob and segment it with Kmeans
        mask_container = np.zeros((image_height, image_width), dtype=np.uint8)
        print(mask_container.shape)

        array_segmented_moles = []
        array_cropped_moles = []
        infos = []
        mole_number = 0
        for i, keypoint in enumerate(keypoints):
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            r = int(keypoint.size / 2)
            crop = self.image[max(y - r * 2, 0):min(y + r * 2, image_height),
                   max(x - r * 2, 0):min(x + r * 2, image_width), :]

            if crop.size > 0:
                # dato che la blob mi identifica anche gli sticker, controllo se il keypoint corrisponde allo sticker (vedo la maschera degli sticker)
                # se corrisponde a 255 (valore dove è stato segmentato lo sticker) allora non faccio niente e passo al prossimo
                if self.sticker.sticker_mask[y, x] != 255:  # se il pixel che mi ha dato la blob è pixel verde dello sticker allora non analizzo, altrimenti si

                    mask_opened = self.segment_kmeans(crop)

                    # TODO ripulire immagine con morphology e label, controllo anche che sia un potenziale neo andando a vedere se
                    # l'area del kmeans è < del numero di pixels a False
                    is_cleaned_segmentation = self.cleaning_segmented_image(mask_opened)

                    if is_cleaned_segmentation:

                        mole_number += 1

                        # riempio buchi e cose varie

                        # LEVO LE VARIE ISOLETTE SPARSE NON CENTRALI
                        # Immagine binaria con le regioni individuate
                        labeled_image = measure.label(mask_opened)

                        # Trova le proprietà delle regioni
                        #regions = measure.regionprops(labeled_image)

                        # Trova l'indice della regione che contiene il pixel centrale
                        height, width = labeled_image.shape
                        center_pixel = (height // 2, width // 2)
                        region_index = labeled_image[center_pixel] - 1

                        # Crea un'immagine con solo la regione del pixel centrale
                        filtered_image = np.zeros_like(mask_opened)
                        filtered_image[labeled_image == region_index + 1] = 255


                        # incollo le singole maschere nella maschera contenitore
                        mask_container[max(y - r * 2, 0):min(y + r * 2, image_height), max(x - r * 2, 0):min(x + r * 2, image_width)] = filtered_image

                        # salvo i singoli nei segmentati
                        array_segmented_moles.append(filtered_image)

                        # salvo l'immagine croppata
                        array_cropped_moles.append(crop)

                        # info single mole
                        mole_info = self.mole_analysis(filtered_image)
                        infos.append(mole_info)



        return array_cropped_moles, array_segmented_moles, mask_container, infos


    def blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 20
        params.maxThreshold = 255

        # Filter by Area
        params.filterByArea = True
        params.minArea = 150
        params.maxArea = self.image.shape[0] * self.image.shape[1]

        params.minCircularity = 0.1
        params.minConvexity = 0.1
        params.minInertiaRatio = 0.1

        return cv2.SimpleBlobDetector_create(params)

    def segment_kmeans(self, crop):
        # Segment the crop with Kmeans
        # Esegui il KMeans con i centroidi iniziali personalizzati
        kmeans = KMeans(n_clusters=2)
        pixels = np.float32(crop.reshape(-1, 3))
        kmeans.fit(pixels)
        labels = kmeans.labels_.reshape(crop.shape[:2])

        mask = np.uint8(labels == 1) * 255

        # Applica la chiusura per riempire i buchi all'interno delle regioni
        mask_closed = binary_closing(mask, disk(2))

        # Applica l'apertura per eliminare i pixel sparsi al di fuori delle regioni
        mask_opened = binary_opening(mask_closed, disk(2))
        # Inverti i valori della maschera se i pixel negli angoli sono tutti 1
        if mask_opened[0, 0] == mask_opened[0, -1] == mask_opened[-1, 0] == mask_opened[-1, -1] == 1:
            # così ho che i nei sono identificati con pixel = 1
            mask_opened = ~mask_opened

        return mask_opened

    def cleaning_segmented_image(self, segmented_image):
        # Conteggio dei pixel a True
        true_count = np.count_nonzero(segmented_image)

        # Conteggio dei pixel a False
        false_count = segmented_image.size - true_count

        if true_count > false_count:
            return False
        else:
            # procedi con il salataggio delle info del neo
            return True


    def mole_analysis(self, mole_mask):
        """
        Analizzo le maschere e i rispettivi nei uno alla volta in
        modo da salvare poi le informazioni neo-info_neo a coppie. Più facile da salvare nel DB forse
        :param array_cropped_moles: lista di array2D che rappresentano le immagini originali dei nei singoli
        :param array_cropped_moles: lista di array2D che rappresentano le immagini binarie dei nei singoli
        :return: lista di dataframe, in ogni dataframe ho i diversi nei e presenti in un singolo crop (idealmente 1) e le rispettive aree
        """

        # rietichetto le regioni
        filtered = measure.label(mole_mask)
        # calcola le proprietà di tutte le regioni individuate
        props = measure.regionprops_table(filtered, properties=['label', 'area', 'equivalent_diameter', 'solidity', 'centroid',
                                                      'major_axis_length', 'minor_axis_length'])

        # area neo più grande
        area = props['area'][0]

        return area

    def prepara_risultati_flutter(self, mask_container):
        """
        :param risultati_analisi:
        :return:
        """

        # Etichetta le regioni connesse nella maschera contenitore
        labeled_mask = measure.label(mask_container)

        # Crea un'immagine RGB in cui le regioni sono colorate in modo diverso
        regions_color = color.label2rgb(labeled_mask, image=self.image)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Disegna l'immagine originale sull'asse
        ax.imshow(self.image)

        # Sovrapponi l'immagine delle regioni sull'immagine originale
        ax.imshow(regions_color, alpha=0.7)

        # Nascondi gli assi
        ax.axis('off')

        # Mostra la figura
        plt.show()
        # Prepara i risultati dell'analisi in un formato adatto per l'applicazione Flutter
        # Restituisci i risultati preparati
        return regions_color

    def manda_risultati_a_flutter(self, risultati_flutter):
        """

        :param risultati_flutter:
        :return:
        """
        # Invia i risultati preparati all'applicazione Flutter utilizzando un metodo di comunicazione (es. API, socket, ecc.)
        pass

    def convert_pixel2mm(self, infos):
        predicted_areas = []
        for i in range(len(infos)):
            predicted_area = (name.AREA_STICKER * infos[i]) / self.sticker.area_sticker_pixel
            predicted_areas.append(predicted_area)

    def calculate_growth_rate(self):
        """
        Assumo che in input all'oggetto della classe venga data:
            una maschera binaria segmentata di una parte del corpo (OLD MASK)
            l'immagine nuova a colori appena scattata
            aree in pixel e in mm calolati precedentemente
        L'immagine nuova a colori viene

        :return:
        """