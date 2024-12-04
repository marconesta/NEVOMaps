from moles_identification import NevoMaps
from sticker_identification import Sticker
from skimage import io


PATH = "/Users/alessandroclemente/Desktop/Int. Project/NevoMaps/Foto with moles/Test_finale_stickers.jpg"




if __name__ == '__main__':



    image = io.imread(PATH)

    sticker = Sticker(image)

    measure = sticker.area_sticker
    mask = sticker.sticker_mask

    s = NevoMaps(img=image, sticker_obj=sticker)
    array_cropped_moles, array_segmented_moles, mask_container, infos = s.analizza_immagini()
    print()

