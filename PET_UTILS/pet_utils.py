import dicom
import numpy as np
import os


# Load CT scan
def load_dicom_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
def np_array_from_dicom_list(dcm_list):
    return np.stack([item.pixel_array for item in dcm_list],  axis=0)


def remove_ct_background(ct_image):
    ct_image[ct_image<0]=0
    return ct_image

def extent_from_dicom(dcm_list):
    return [0, len(dcm_list)*dcm_list[0].SliceThickness, 0, dcm_list[0].pixel_array.shape[0]*dcm_list[0].PixelSpacing[0], 0, dcm_list[0].pixel_array.shape[1]*dcm_list[0].PixelSpacing[1]]

def sirf_reg_to_numpy(image):
    if image.ndim is 3:
        return np.transpose(np.flip(image,axis=2),(2,0,1))
    elif image.ndim is 5:
        return np.transpose(np.flip(image,axis=2),(2,0,1,3,4))

