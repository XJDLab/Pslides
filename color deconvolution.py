import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hed, hed2rgb
from skimage import io
from skimage.exposure import rescale_intensity
import colorsys
from PIL import Image

# deconvolution of the hematoxylin and DAB channel from ihc images
def ihc2hema (file):  
    img=io.imread(file)
    img_hed=rgb2hed(img)
    null = np.zeros_like(img_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    return ihc_h, ihc_d

# rescale and generate pesudocolor images form hematoxylin and DAB channels

h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))

# Cast the two channels into an RGB image, as the blue and green channels, respectively
zh = np.dstack((null, null, h)) # give a blue fluorescence look of the nucleus like DAPI staining
zd_r=np.dstack((d, null, null)) # give a red fluorescence look of the DAB positive staining
zd_g=np.dstack((null, d, null)) # give a green fluorescence look of the DAB positive staining




## to generate more pesudocolors from the generated red or green fluorescence look of the DAB positive staining


filename = #path to the image

image = Image.open(filename).convert('RGB')

image.load()
r, g, b = image.split()
result_r, result_g, result_b = [], [], []

for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
    h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
    rgb = colorsys.hsv_to_rgb(h-0.1, s, v) # the value 0.1 can be adjusted to generate different pesudocolors
    pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
    result_r.append(pixel_r)
    result_g.append(pixel_g)
    result_b.append(pixel_b)


r.putdata(result_r)
g.putdata(result_g)
b.putdata(result_b)


image = Image.merge('RGB', (r, g, b))
image.save(r"destination path to the image")