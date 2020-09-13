import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dicom
import pet_utils as petut
from matplotlib.animation import FuncAnimation
import time
import os
def imshow(image, limits=[], title='',colormap="viridis",extent=None,alpha=1,colorbar="off"):
    """Display an image with a colourbar, returning the plot handle. 
    
    Arguments:
    image -- a 2D array of numbers
    limits -- colourscale limits as [min,max]. An empty [] uses the full range
    title -- a string for the title of the plot (default "")
    colormap -- a string with a matplotlib colormap (default "viridis")
    """
    plt.title(title)
    bitmap=plt.imshow(image,cmap=colormap,extent=extent,alpha=alpha)
    if len(limits)==0:
        limits=[np.nanmin(image),np.nanmax(image)]
    plt.clim(limits[0], limits[1])
    if colorbar is "on":
        plt.colorbar(shrink=.6)
    plt.axis('off')
    fig=plt.gcf()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    return bitmap


def dvf_show(dvf_arr,axis=0,mode="components",colormap=None,title=None,limits=None):
    """Display vector fields (tuned for displacement) with a colourbar

    
    Arguments:
    dvf_arr -- a 5D (nifti format) array of numbers. the 4th dimension should be shape==1
    slice -- the slice that you want to visualize. It slices trhough the leftmost (x) dimension. (default is the mid slice)
    mode -- either "slices" or "norm". "slices" creates X,Y,Z slices cuts, while "norm" computes the absolute value of the vectors. (default "slices")
    colormap -- a string with a matplotlib colormap (default "RdBu" for slices, "viridis" for norm)


    Colourbar limits for "slice" are centered at zero with max(abs(minval),abs(maxval)) as upper and lower limits and [minval, maxval] for norm

    """
   
    if colormap is None:
        if mode is "components":
            colormap="RdBu"
        else:
            colormap="viridis"
    if len(dvf_arr.shape) is not 5:
        raise "dvf_arr does not have 5 dims"

    
    if axis==1:
        dvf_arr=np.transpose(dvf_arr,(1,0,2,3,4))
    if axis==2:
        dvf_arr=np.transpose(dvf_arr,(2,1,0,3,4))


    if mode is "components":
        if title is None:
            if axis==1:
                title=["x (ANT-POS)","y (Left-Right)","z (CRA-CAU)"]
        dvf=[dvf_arr[:,:,:,0,0].squeeze(), dvf_arr[:,:,:,0,1].squeeze(), dvf_arr[:,:,:,0,2].squeeze()]
        if limits is None:
            limits=[]
            for i in range(3):
                m=np.max(np.abs([np.nanmin(dvf[i]), np.nanmax(dvf[i])]))
                limits.append([-m, m])

        _imshow3D_display_(dvf,limits=limits,colormap=colormap,title=title)
       
    if mode is "norm":
        if title is None:
            title="Norm"
        dvf_norm= np.linalg.norm(dvf_arr,axis=4).squeeze()
        _imshow3D_display_([dvf_norm],title=title,colormap=colormap,limits=limits)


def MIP(image, axis=0):
    """Computes a maximum intensity projection along desired axis

    Arguments:
    image -- A 3D image
    axis -- axis on which to take the MIP (default 0)
    """
    return np.max(image,axis=axis).squeeze()

def imshow3D(image,axis=0,**kwargs):
    """imshow3D visualizes 3D images with a slider

    Argumets
    image -- a 3D image or list of 3D images (can be dicom.dataset.FileDataset or a list of 3D np.arrays)
    **kwargs -- see imshow() arguments

    title has to be a list if image is a list. 
    """

    ## Parse input
    # Handle extent:
    if "extent" not in kwargs:
        kwargs["extent"]=None
    kwargs["extent"]=_get_extent_(image,axis,kwargs.pop("extent"))

    n_images=1
    # if we got more than 1 image. 
    if isinstance(image,list):
        # This could mean 2 things: a list of dicom images, yet just 1  volume, or more than 1 volume. 
        if isinstance(image[0],np.ndarray):
            n_images=len(image)
        elif isinstance(image[0],dicom.dataset.FileDataset):
            n_images=1
            image=[petut.np_array_from_dicom_list(image)]
    else:
        image=[image]

        
    # Make sure that if multiple inputs, they are the same size
    for i in range(n_images-1):
        if kwargs["extent"] is None:
            assert image[i].shape == image[-1].shape, "Images should be the same shape or a extent given" 
    # Permute if needed
    for i in range(n_images):
        if axis==1:
            image[i]=np.transpose(image[i],(1,0,2))
        if axis==2:
            image[i]=np.transpose(image[i],(2,0,1))

    if "animate" not in kwargs:
        kwargs["animate"]=False


    if kwargs["animate"] is False:
        _imshow3D_display_(image,**kwargs)
    else:
        _imanimate_(image,**kwargs)

def _get_extent_(image,axis,extent):
    '''
    This function handles all the possible inputs for the extent optional argument.
    '''
    # If its dicom:
    if extent is not None:
        assert len(extent)==6, "extent should be 6 valued array"
        if axis==0:
            extent=extent[4:6] + extent[2:4]
        if axis==1:
            extent=extent[4:6] + extent[0:2]
        if axis==2:
            extent=extent[2:4] + extent[0:2]
    elif isinstance(image,list) and isinstance(image[0],dicom.dataset.FileDataset):
        if extent is None:
            if axis==0:
                extent=[0, image[0].pixel_array.shape[1]*image[0].PixelSpacing[1],0, image[0].pixel_array.shape[0]*image[0].PixelSpacing[0]]
            if axis==1:
                #np.transpose(image[i],(1,0,2))
                extent=[0, image[0].pixel_array.shape[1]*image[0].PixelSpacing[1],0, len(image)*image[0].SliceThickness]
            if axis==2:
                #np.transpose(image[i],(2,0,1))
                extent=[ 0, image[0].pixel_array.shape[0]*image[0].PixelSpacing[0],0, len(image)*image[0].SliceThickness]
    else:
        extent=None
    return extent
def _imshow3D_display_(image,**kwargs):
    '''
    this is the one that actually displays
    image must be a list of np.arrays
    '''
    colormap, limits, title, extent, colorbar, _ = _unpack_disp_kwargs_(image,**kwargs)
    # Start the plotting
    n_images=len(image)

    fig=plt.gcf()
    fig.subplots_adjust(left=0.25, bottom=0.25)
    l=[]
    for i in range(n_images):
        plt.subplot(1,n_images,i+1)
        l.append(imshow(image[i][image[i].shape[0]/2],colormap=colormap[i],limits=limits[i],extent=extent,title=title[i],colorbar=colorbar[i]))

    ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    max_val=np.max([im.shape[0] for im in image])-1

    slidx = Slider(ax, 'slice', 0,max_val, valinit=max_val/2, valfmt='%d')
    _imshow3D_display_._slider=slidx
    def update(val):
        idx = slidx.val
        for i in range(n_images):
            l[i].set_data(image[i][int(idx/max_val*(image[i].shape[0])-1)])
        fig.canvas.draw_idle()
    slidx.on_changed(update)

def _imshow3D_display_overlay_(image,alpha,**kwargs):
    '''
    this is the one that actually displays
    image must be a len(2) list of np.arrays
    alpha must be a len(2) list of float
    '''
    colormap, limits, title, extent, _, _ = _unpack_disp_kwargs_(image,**kwargs)

    assert isinstance(alpha,list) and len(alpha)==2, "alpha has to be a length 2 list"
    
    # Start the plotting
    n_images=len(image)
    assert n_images==2, "Only 2 overlayed images supported"

    fig=plt.gcf()
    #fig.subplots_adjust(left=0.25, bottom=0.25)
    l=[]
    for i in range(n_images):
        #plt.subplot(1,n_images,i+1)
        l.append(imshow(image[i][image[i].shape[0]/2],colormap=colormap[i],limits=limits[i],extent=extent,title=title,alpha=alpha[i]))

    ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    max_val=np.max([im.shape[0] for im in image])-1

    slidx = Slider(ax, 'slice', 0,max_val, valinit=max_val/2, valfmt='%d')
    _imshow3D_display_._slider=slidx
    def update(val):
        idx = slidx.val
        for i in range(n_images):
            l[i].set_data(image[i][int(idx/max_val*(image[i].shape[0])-1)])
        fig.canvas.draw_idle()
    slidx.on_changed(update)

def imshow3d_petct(ct,pet,axis=0,alpha=[1,0.6],colormap=['gray','afmhot'],**kwargs):

    if "extent" not in kwargs:
        kwargs["extent"]=None
    kwargs["extent"]=_get_extent_(ct,axis,kwargs.pop("extent"))
     # if we got more than 1 image. 
    if isinstance(ct,list) and isinstance(ct[0],dicom.dataset.FileDataset):
            ct=petut.np_array_from_dicom_list(ct)
    image=[ct,pet]

    # Permute if needed
    n_images=2
    for i in range(n_images):
        if axis==1:
            image[i]=np.transpose(image[i],(1,0,2))
        if axis==2:
            image[i]=np.transpose(image[i],(2,0,1))

    _imshow3D_display_overlay_(image,alpha=alpha,colormap=colormap,**kwargs)




def _imanimate_(image,**kwargs):
    '''
    this is the one that actually displays and produces the animation
    image must be a list of np.arrays
    '''
    
    colormap, limits, title, extent, colorbar, filename = _unpack_disp_kwargs_(image,**kwargs)
    # Start the plotting
    fig=plt.gcf()
    #maximize window
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    max_val=np.max([im.shape[0] for im in image])-1
    
    def update(val):
        n_images=len(image)
        fig=plt.gcf()
        max_val=np.max([im.shape[0] for im in image])-1
        # initaliation
        if val is 0:
            fig.subplots_adjust(left=0.25, bottom=0.25)
            l=[]
            ax=[]
            for i in range(n_images):
                ax.append(plt.subplot(1,n_images,i+1))
                l.append(imshow(image[i][image[i].shape[0]/2],colormap=colormap[i],limits=limits[i],extent=extent,title=title[i],colorbar=colorbar[i]))
            update._l=l
            update._ax=ax
        # normal plotting
        for i in range(n_images):
            update._l[i].set_data(image[i][val])
            update._ax[i].set_title(title[i]+ " "+ str(val+1)+"/"+str(max_val+1))
        fig.canvas.draw_idle()
    ani = FuncAnimation(fig, update, frames=max_val+1)
    _imanimate_._ani=ani
    if filename:
        if not os.path.isdir(os.path.abspath(os.getcwd())+'/animation/'):
            os.mkdir(os.path.abspath(os.getcwd())+'/animation/')
        ani.save(os.path.abspath(os.getcwd())+'/animation/'+filename, writer='imagemagick', fps=1)
 

def _unpack_disp_kwargs_(image,**kwargs):
    # Unpack kwargs (in case there are various)
    if "colormap" in kwargs:
        colormap=kwargs.pop("colormap")
        if not isinstance(colormap,list):
            colormap=[colormap]
            for _ in range(len(image)-1):
                colormap.append(colormap[0])
        assert len(colormap)==len(image), "The list of colormaps has to be the same length as list of images, or 1"
    else:
        colormap=[None]*len(image)
    
    if "limits" in kwargs:
        limits=kwargs.pop("limits")
        if isinstance(limits[0],int): # if the first instance is just a number, then the input has only been a list and not a nested list
            limits=[limits]
            for _ in range(len(image)-1):
                limits.append(limits[0])
        assert len(limits)==len(image), "The list of limits has to be the same length as list of images, or 1"
    else:
        limits=[[]]*len(image)

    if "title" in kwargs:
        title=kwargs.pop("title")
        if not isinstance(title,list): 
            title=[title]
            for _ in range(len(image)-1):
                title.append(title[0])
        assert len(title)==len(image), "The list of title has to be the same length as list of images, or 1"
    else:
        title=[""]*len(image)

    if "extent" in kwargs:
        extent=kwargs.pop("extent")
    else:
        extent=None

    if "colorbar" in kwargs:
        colorbar=kwargs.pop("colorbar")
        if not isinstance(colorbar,list): 
            colorbar=[colorbar]
            for _ in range(len(image)-1):
                colorbar.append(colorbar[0])
        assert len(colorbar)==len(image), "The list of limits has to be the same length as list of images, or 1"
    else:
        colorbar=["on"]*len(image)
        
    if "filename" in kwargs:
        filename=kwargs.pop("filename")
    else:
        filename=None

    return colormap, limits, title, extent, colorbar,filename
    