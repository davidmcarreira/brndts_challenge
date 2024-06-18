import os
import cv2

def load_image(path, color_map = 'rgb'):
    """
    
    Helper function that loads the images and changes their color if needed.
    
    Args:
        - path (str): Where to look for the image. It can be a single image or a directory.
    Returns:
        - image (numpy.ndarray): Yields the image to be further processed.
    
    """
        
    if os.path.isfile(path):
        print(f'>> Processing a single image: {path}')
        image = cv2.imread(path)
    
        if color_map.lower() == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Inverts the default OpenCV's channel notation to the usual RGB
            
        yield image
        
    elif os.path.isdir(path):
        print(f'>> Processing a directory of images: {path}')
        for root, dirs, files in os.walk(path): #Iterates over the directory
            for file in files:
                image_path = os.path.join(root, file) #And extracts only the path to the images
                image = cv2.imread(image_path)
    
                if color_map.lower() == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yield image
                
    else:
        raise ValueError(f'>> {path} is not a single file or directory.')