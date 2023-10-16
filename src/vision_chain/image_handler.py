'''
Burhan Qaddoumi
2023-07-05
'''

import cv2 as cv
import numpy as np
from pathlib import Path

CV_RESIZE = [cv.INTER_NEAREST, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_AREA, cv.INTER_LANCZOS4, cv.INTER_LINEAR_EXACT, cv.INTER_MAX]

def get_unique_colors(color_image:np.ndarray):
    """Returns array of unique colors from input 3-channel image array"""
    assert color_image.shape[-1] == 3 and color_image.ndim == 3, f"Input image must be 3-channel color image"
    pixels = color_image.reshape(-1,3)
    
    return np.unique(pixels,axis=0)

class ImageObj:
    """Class to load and manipulate images read from file
    
    Attributes
    ---
    input_path : ``str``
      Input path as string
      
    path : ``pathlib.Path``
      Path object from `input_path`
      
    image : ``np.ndarray``
      Opened image
      
    unique_colors : ``np.ndarray``
      Array of unique colors, as ``tuple`` found in image.
    
    img_hash : ``np.ndarray``
      Array of image hash values calculated using ColorMomentHash (used to compare for similarity to other images)
    
    BGR : ``np.ndarray``
      BGR color space copy of `self.image`, `None` by default unless created from color space conversion
    
    RGB : ``np.ndarray``
      RGB color space copy of `self.image` 
    
    HSV : ``np.ndarray``
      HSV color space copy of `self.image`, `None` by default unless created from color space conversion
    
    GRAY : ``np.ndarray``
      Gray-scale color space copy of `self.image`, `None` by default unless created from color space conversion
    
    width : ``int``
      Image width in pixels
    
    height : ``int``
      Image height in pixels
    
    channels : ``int``
      Number of image channels
    
    Properties
    ---
    colorspace : ``str``
      Current color space of `self.image`
    
    valid_colorspaces : ``list``
      Color space conversions available for `self.image` instance
    
    Methods
    ---
    get_image()
      Called during `__init__` to open image file from path specified
    
    image_recolor()
      Used to modify color space of `self.image`
    
    image_resize()
      Returns resized (copy) of `self.image`, not stored
    """
    def __init__(self,
                path:str|Path,
                is_gray:bool=False, # True if image should load as gray-scale
                ):
    self.input_path = str(path).replace('\\','/')
    self.path = Path(path) # NOTE Could pull additional info from Path object
    self.GRAY = self.HSV = self.RGB = self.BGR = None
    self.__source_img = None
    self.__is_gray = is_gray
    self.image = self.get_image(gray=is_gray)

    # Methods
    def _get_source_image(self,source_img):
        """Only runs on first call of get_image() to populate (private-ish) attribute for source image"""
        if not self._ImageObj__source_img:
            self._ImageObj__source_img = np.copy(source_img)
            self.__source_color = getattr(self,'colorspace')
            self.img_hash = cv.img_hash.colorMomentHash(self._ImageObj__source_img)
        
    def get_image(self,gray=False):
        """Reads image file into class, default read file using BGR color"""
        img = cv.imread(str(self.path)) if not gray else cv.imread(str(self.path),cv.IMREAD_GRAYSCALE)
        self.channels = img.shape[-1] if img.ndim == 3 else 1
        self.height, self.width = img.shape[:2]
        
        if self.channels == 3:
            self.BGR = np.copy(img)
        else:
            self.GRAY = np.copy(img)
            
        self.__colorspace = 'BGR' if self.BGR is not None and not gray else 'GRAY'
        self._get_source_image(img)
        self.unique_colors = get_unique_colors(img) if not gray else np.array([-1,-1,-1]) # TODO add logging message
    
    def image_recolor(self,new_color:str,main:bool=False):
        """
        Usage
        ---
        Convert color space of image to the specified (valid) color space.
        
        Arguments
        ---
        new_color : ``str``
            Color space to convert image to, only allowed values are in `self.valid_colorspaces`
        
        main : ``bool`` = False
            OPTIONAL, will convert `self.image` to new color space when `True` otherwise populates `self.{new_color}` using all uppercase characters.
        
        Returns
        ---
        Nothing returned, only object instance attributes or properties modified.
        """
        assert isinstance(new_color,str) and new_color.upper() in self.valid_colorspaces, f"Argument `new_color` must be of type `str` and be one of {self.valid_colorspaces}"
        
        new_color = new_color.upper()
        curr_color = getattr(self, '_ImageObj__colorspace')
        processing_needed = curr_color != new_color or getattr(self,new_color) is None
        
        if processing_needed:
            recolor_image = getattr(self,new_color)
            combo = (self._ImageObj__source_color,new_color)
            
            if recolor_image is None:
                if self._ImageObj__source_color != 'GRAY':
                    recolor_image = cv.cvtColor(np.copy(self._ImageObj__source_img),self._color_dict[combo])
                    setattr(self,new_color,recolor_image)
                    
                elif self._ImageObj__source_color == 'GRAY' and new_color == 'HSV' and self.BGR is None:
                    self.BGR = cv.cvtColor(np.copy(self._ImageObj__source_img),cv.COLOR_GRAY2BGR)
                    self.HSV = cv.cvtColor(np.copy(self.BGR),cv.COLOR_BGR2HSV_FULL)
                
                elif self._ImageObj__source_color == 'GRAY' and new_color == 'HSV' and self.BGR is not None:
                    self.HSV = cv.cvtColor(np.copy(self.BGR),cv.COLOR_BGR2HSV_FULL)
                    
            if main:
                setattr(self, 'image', np.copy(getattr(self,new_color)))
                self._ImageObj__colorspace = new_color
    
    def image_resize(self,
                     dims:tuple|list,
                     interp=cv.INTER_CUBIC
                    ):
        """
        Usage
        ---
        
        Arguments
        ---
        dims : ``list`` | ``tuple``
            Iterable composed of all ``int`` or ``float`` values (width, height) that are greater than zero. Resizes by absolute value if both values are ``int`` or by using ratio if both values are ``float`` type.
            
        interp : ``int`` = cv.INTER_CUBIC
            OPTIONAL, method for pixel interpolation when resizing, see OpenCV documentation for addtional details.
        
        Returns
        ---
        Copy of resized `self.image` based on the type of `dims` values, if values are unmatched or invalid, no resizing is performed and image is returned with original size.
        """
        assert all([i > 0 for i in dims]), f"Image resizing dimensions must all be integer or float types with values greater than 0."
        assert interp in CV_RESIZE, f"Resizing interpolation method must be an option from {CV_RESIZE}."
        
        if all([type(i) == int for i in dims]):
            resize_args = dims, 0, 0, interp

        elif all([type(i) == float for i in dims]):
            resize_args = None, None, *dims, interp
            
        else:
            resize_args = None, None, 1.0, 1.0, None
            
        return cv.resize(np.copy(self.image),*resize_args)
    
    ## Properties
    @property
    def valid_colorspaces(self):
        self._color_dict = {
                            ('BGR','RGB'):cv.COLOR_BGR2RGB,
                            ('BGR','GRAY'):cv.COLOR_BGR2GRAY,
                            ('BGR','HSV'):cv.COLOR_BGR2HSV_FULL,
                            ('BGR','BGR'):None,
                            ('GRAY','BGR'):cv.COLOR_GRAY2BGR,
                            ('GRAY','RGB'):cv.COLOR_GRAY2RGB,
                           }
    return sorted(set([i for k in self._color_dict.keys() for i in k]))
      
    @valid_colorspaces.setter
    def valid_colorspaces(self,anything):
        pass
      
    @property
    def colorspace(self):
        """Color space of `self.image`, can be directly modified to change color space"""
        return self._ImageObj__colorspace
      
        @colorspace.setter
        def colorspace(self,x:str):
            assert x.upper() in self.valid_colorspaces, f"Valid colorspaces are {self.valid_colorspaces}"
            
            getattr(self,'image_recolor')(new_color=x.upper(),main=True)
            
            self._ImageObj__colorspace = x.upper()
          
