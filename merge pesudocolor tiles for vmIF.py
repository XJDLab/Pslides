import numpy as np
from skimage import data
from skimage import io

## an example of generate biplex images
img_dapi=io.imread(r"path to the blue channel image")
img_red=io.imread(r"path to the red channel image")
img_merge=abs(img_dapi/255.0-img_red/255.0)
io.imsave(r"path to the merged image",img_merge)
## this procedure can be repeated to generate multiplex virtual IF images