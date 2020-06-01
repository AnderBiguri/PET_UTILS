# PET_UTILS
Bunch of useful functions for PET data.

**Very work in progress**, *caveat utilitor*. 

Accepts np.arrays or dicom files loaded with pydicom. 

Can do: 

 - `imshow` 2D images
 - `imshow3D` 3D images with sliders. Accepts lists of np.arrays, a np.array or a list of dicom slices from pydicom.
 - `imshow3D_petct` Overlayed PET/CT image with sliders. 
 - `dvf_show` displays displacement fields, input as 5D np.array. Can do x,y,z or norm plots. 
