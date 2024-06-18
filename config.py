MODEL_PATH = 'brndts_challenge/models/yolov8s-oiv7.pt' #Path to the detection model
SEG_MODEL_PATH = 'brndts_challenge/models/yolov8s-seg.pt' #Path to the segmentation model
DEVICE = 'gpu' #Where the iamges are processed

#Where the images are stored
SOURCE = 'brndts_challenge/source/image.png' 
ADVERT = 'brndts_challenge/source/advert.jpeg'

#_edge_detection(image, alpha, beta, sp, cr, t, set_px)
ALPHA = 2.5 #Contrast adjustment
BETA = 0 #Brightness adjustment
SP = 15 #Spatial windows radius for the noise reduction
CR = 21 #Spatial color radius for the noise reduction
T = 100 #Pixels higher that this threshold are set to SET_PX (lower are set to 0)
SET_PX = 255 #Final value of the thresholded pixels.

#advert_locations(self, image, conf_score):
CONF_SCORE = 0.001 #Detection confidence score

#segment_occlusions(self, image, seg_conf_score)
SEG_CONF_SCORE = 0.5 #Segmentation confidence score

#gen_advert(advert, image, cnt_list, seg_occlusion_masks, warp_boxes, down_scale, border_t, border_color, warp, bordered, mode, saveE)
DOWN_SCALE = 0.95 #Factor that reduces the image size to account for the borders
BORDER_T = 0.03 #Border thickness
BORDER_COLOR = (0,0,0) #Tuple with the color of the border
WARP = False  #Boolean that determines if the advert suffers a perspective warp or not
BORDERED = True #Boolean that determisn if a border is added
COLOR_CORRECTION = 'histogram' #histogram' (matches the histogram of the advertisement with the background) or 'color_transf' (adjusts the colors using the mean and standard deviation)
SMOOTHING = 0.9 #The amount of smoothing applied to the picture.
MODE = 'area' #'area' looks for the bigger box to insert the ad, 'total' places the ad everywhere it can be and 'random' is self explanatory 
SAVE = True #If true saves the iamge in the cwd

