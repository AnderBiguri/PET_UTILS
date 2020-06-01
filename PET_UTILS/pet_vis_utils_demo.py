import sirf.Reg as reg
import matplotlib.pyplot as plt
import numpy as np

import pet_vis_utils as petvis



# Load 2 3D images:
my_data_path=""
img1=reg.ImageData(my_data_path)
img2=reg.ImageData(my_data_path2)
img1=img1.as_array()
img2=img2.as_array()
#Load a displacement vector field
dvf=reg.NiftiImageData3DDisplacement(my_dvf_pat)
dvf=dvf.as_array()


# Plot MIP
plt.figure()
plt.subplot(1,2,1)
petvis.imshow(np.flip(petvis.MIP(img1),axis=0),limits=[], title= 'Image 1',colormap="Greys")
plt.subplot(1,2,2)
petvis.imshow(np.flip(petvis.MIP(img2),axis=0),limits=[0, 1], title='Image 2',colormap="Greys")
plt.show()

# Plot 3D image
plt.figure()
petvis.imshow3D([img1,img2],limits=[0, 5e4], title=['Image 1', 'Image 2'],colormap="Greys")
# alternatively: imshow3D(img1,limits=[0, 5e4], title='Image 1',colormap="Greys")
plt.show()

# Plot DVf:

plt.figure()
petvis.dvf_show(dvf,slice=128,mode="slices")
plt.show()

# or just norm:
plt.figure()
petvis.dvf_show(dvf,slice=128,mode="norm")
plt.show()


