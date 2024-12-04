
from cmath import sqrt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from PIL import Image, ImageEnhance, ImageDraw
import random
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, measure, exposure, color
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.morphology import binary_closing, binary_opening, disk

"""
In questa prima parte vogliamo fare in modo che la foto dello user, scattata a distanza di tempo, possa essere più 
uguale possibile alla prima foto scattata in modo da poter fare un confronto più preciso
In this first part we want to make sure that the user's photo, taken at a distance of time, 
can be as similar as possible to the first photo taken so that we can make a more precise comparison
comment
"""
def match_images(img1, img2):
    """

    :param img1:
    :param img2:
    :return:
    """

    # Converte le immagini in scala di grigi
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Rileva i punti di interesse (features) nelle immagini
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Trova le corrispondenze tra i punti di interesse nelle due immagini
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Filtra le corrispondenze usando il ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Estrae le coordinate dei punti di interesse nelle due immagini
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calcola la trasformazione prospettica tra le due immagini
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Applica la trasformazione alla prima immagine
    h, w = gray1.shape
    aligned_img1 = cv2.warpPerspective(img1, M, (w, h))

    # Visualizza le immagini allineate
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    return aligned_img1


def masking(img1, img2):
    """

    :param img1:
    :param img2:
    :return:
    """

    # leggi l'immagine binaria
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # definisci il kernel per l'operazione di opening e closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # applica l'operazione di binary opening
    opening_1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    opening_2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    # applica l'operazione di binary closing
    closing_1 = cv2.morphologyEx(opening_1, cv2.MORPH_CLOSE, kernel)
    closing_2 = cv2.morphologyEx(opening_2, cv2.MORPH_CLOSE, kernel)

    threshold_value = 90 
    ret, im1_mask = cv2.threshold(closing_1, threshold_value, 255, cv2.THRESH_BINARY)
    ret, im2_mask = cv2.threshold(closing_2, threshold_value, 255, cv2.THRESH_BINARY)

    return im1_mask, im2_mask

def overlap_images(im1, im2):
    im1= Image.fromarray(im1)
    im2= Image.fromarray(im2)
    diff = ImageChops.difference(im1, im2)
    diff.show()
    return diff



# Carica le due immagini
img1 = cv2.imread("C:/Users/paola/OneDrive/Desktop/NevoMaps/moles_detection/compare_two_images/mano_1.jpeg")
img2 = cv2.imread("C:/Users/paola/OneDrive/Desktop/NevoMaps/moles_detection/compare_two_images/mano_2.jpeg")

aligned_img1= match_images(img1, img2)


#CONFRONTO LE DUE IMMAGINI
im1_mask, im2_mask = masking(aligned_img1, img2)

diff = overlap_images(im1_mask, im2_mask)

image = np.array(diff)


# PER POTER CAPIRE SE CI SONO NUOVI NEI, DOPO AVER SOVRAPPOSIZIONATO LE IMMAGINI, POSSO SPOTTARE I NUOVI NEI 

# Setup SimpleBlobDetector parameters
def blob_for_new_moles():
    """

    :return:
    """
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 100
    params.filterByColor = True
    params.blobColor = 255  #255 -->> light blobs

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 7000

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)
    print("Number of differences detected:", len(keypoints))
    # in bianco e nero
    img_with_new_moles = cv2.drawKeypoints(image, keypoints,  np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Number of new moles:", len(keypoints))


    # Crea una figura con due subplot
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15))

    # Visualizza le immagini nei subplot
    #ax1.imshow(img_with_new_moles)
    #ax2.imshow(image)

    # Mostra la figura
    #plt.show()
