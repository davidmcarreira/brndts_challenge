import cv2
import os
import torch
import numpy as np

from ultralytics import YOLO
from config import MODEL_PATH, SEG_MODEL_PATH, DEVICE, SOURCE, \
                   ALPHA, BETA, SP, CR, T, SET_PX, \
                   CONF_SCORE, SEG_CONF_SCORE, \
                   DOWN_SCALE, BORDER_T, BORDER_COLOR, WARP, BORDERED, COLOR_CORRECTION, SMOOTHING, MODE, SAVE

class BRNDTS:
    def __init__(self, model_path = MODEL_PATH, seg_model_path = SEG_MODEL_PATH, device = DEVICE):
        
        if device.lower() == 'gpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Redundant verification to guarantee processing
        else:
            self.device = 'cpu'
        
        self.model_path = model_path
        self.seg_model_path = seg_model_path
        
        self.model = YOLO(self.model_path).to(self.device)
        self.seg_model = YOLO(self.seg_model_path).to(self.device)
    
    @staticmethod #Place in utils module      
    def _box_margin(box_coords, margin):
        """
        Adds a margin to the bounding box to include more background.
        This helps the thresholding when looking for possible edges in the potential
        advert placement.

        Args:
            - box_coords (list -> float): Coordinates of the bounding box (x_min, y_min, x_max, y_max).
        Params:
            - margin (int): Margin value to add around the bounding box.

        Returns:
            - x_min, y_min, x_max, y_max (float): Adjusted coordinates with added margin.

        """
        #Iterates over the bouning box's coordinates
        for i, value in enumerate(box_coords):
            if box_coords[i]<margin: #Cases where edge coordiantes can't be reduced (i.e, if they are at the edge of the picture)
                continue
            else:
                if i in [0, 1]: # 0 and 1 are the indexes of the x_min and y_min respectively
                    box_coords[i] = value-margin # Handles the x_max and y_max cases
                else:
                    box_coords[i] = value+margin
        x_min, y_min, x_max, y_max = [int(coord) for coord in box_coords] #List comprehension to satisfy the int requirement of the coordinates

        return x_min, y_min, x_max, y_max
    
    
    
    @staticmethod
    def _edge_detection(image, alpha = ALPHA, beta = BETA, sp = SP, cr = CR, t = T, set_px = SET_PX):
        """
        Detects the edges of the potential ad places. The detection model produces only the Region of Interest (RoI)
        and not the exact location. The edge detection is performed by an automatic thresholding method called Otsu.
        To better distinguish the edges, the contrast and brightness are adjusted and Mean Shift Filtering is applied 
        for noise reduction.

        Args:
            - image (numpy.ndarray): Input image (BGR format).
        Params:
            - alpha (float): Contrast control.
            - beta (float): Brightness control.
            - sp (int): Spatial windows radius for the noise reduction
            - cr (int): Spatial color radius for the noise reduction
            - t (int): Pixels higher that this threshold are set to SET_PX (lower are set to 0)
            - set_px (int): Final value of the thresholded pixels.

        Returns:
            - thresh (numpy.ndarray): Binary image of the edges.
        """
        
        contrast_adjust = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) #Enhances contrast and brightness so the edges are more visible
        blur = cv2.pyrMeanShiftFiltering(contrast_adjust, 15, 21) #Reduces noise with mean shift filtering
        gray = cv2.cvtColor(contrast_adjust, cv2.COLOR_BGR2GRAY) #Converts to grayscale to prepare for edge detection
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #Automatic thresholding using Otsu to detect the edges

        return thresh #Returns the thresholded binary image
    
    
    
    @staticmethod
    def _resize_and_padding(image, box_width, box_height, padding_color=(60, 60, 60)):
        """
        Helper method to resize the advert in accordance with the size of the possible placement boundig box
        without stretching it in any way. To fill the gaps that result from the resizing, colored padding is applied.
        to fill the gaps.
        
        Args:
            - image (numpy.ndarray): Image for the advert.
            - box_width (int): Width of the bounding box.
            - box_height (int): Height of the bounding box.
        Params:
            - padding_color (tuple): Color of the padding that fills the empty spaces.
        Returns:
            - padded_image (numpy.ndarray): Resized and padded ad.
        
        """
        image_height, image_width, _ = image.shape
        aspect_ratio = image_width / image_height #Aspect ratio of the ad

        #Calculates the target dimensions to maintain aspect ratio
        if aspect_ratio > 1:  #Image is wider
            target_width = box_width
            target_height = int(target_width / aspect_ratio)
        else:  #Image is taller or square
            target_height = box_height
            target_width = int(target_height * aspect_ratio)

        #Resizes the image to fit within the target dimensions
        resized_image = cv2.resize(image, (target_width, target_height))

        #Calculate padding needed to center the resized image into the box
        pad_top = max(0, (box_height - resized_image.shape[0]) // 2)
        pad_bottom = max(0, box_height - resized_image.shape[0] - pad_top)
        pad_left = max(0, (box_width - resized_image.shape[1]) // 2)
        pad_right = max(0, box_width - resized_image.shape[1] - pad_left)

        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padding_color)

        return padded_image
    
    
    
    @staticmethod
    def _warp_advert(idx, x, y, advert_height, advert_width, warp_boxes):
        """
        Helper function that applies the computed warps to match the correct perspective.
        
        Args:
            - idx (int): Index of the warp coordinates in the coordinates list.
            - x (int): Left coordinate of the ad.
            - y (int): Top coordinate of the ad.
            - advert_height (int): Height of the advert image.
            - advert_width (int): Width of the advert image.
            - warp_boxes (list): Contains the coordinates for the perpspective warp for every bounding box.
        Returns:
            - advert_warped (numpy.ndarray): Advert image with the perspective warped applied.
        
        """
        #Coordinates of the 4 corners of the image
        top_left = [x, y]
        top_right = [x+advert_width, y]
        bottom_left = [x, y+advert_height]
        bottom_right =[x+advert_width, y+advert_height]
        
        
        src_points = np.float32([top_left, top_right, bottom_left, bottom_right]) #Original corners of the image
        
        #Corners and dimensions of the warped rectangle
        dest_points = np.float32(warp_boxes[idx])
        dest_width = int(warp_boxes[idx][0][0]-warp_boxes[idx][1][0])
        dest_height = int(warp_boxes[idx][0][1]-warp_boxes[idx][1][1])
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dest_points) #Transformation matrix that warps the image
        

        return transform_matrix, dest_width, dest_height
    
    
    
    @staticmethod
    def _color_correction(image, advert, mode):
        """
        Method that applies color correction to the advert with realism in mind.
        It leverages two techniques: histogram equalization and color transfer.
        
        Args:
            - image (numpy.ndarray): Image that serves as the source of the color corrections.
            - advert (numpy.ndarray): Image where the correction is applied.
            - mode (str): "histogram" or "color_transf".
        Returns:
            - corrected (numpy.ndarray): Color corrected image.
        """
    
        #Converts from RGB to LAB space. Because it separates the lightness from the colors, the results are better
        #than opertaing in the RGB color space.
        background = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        foreground = cv2.cvtColor(advert, cv2.COLOR_RGB2LAB)

        #Splitting the image's channels
        bg_l, bg_a, bg_b = cv2.split(background) #Background aka image
        fg_l, fg_a, fg_b = cv2.split(foreground) #Foreground aka advertisement
        
        if mode == 'histogram':
            #Calculating the histograms
            bg_hist, _ = np.histogram(bg_l.flatten(), bins=256, range=(0, 256), density=True)
            fg_hist, _ = np.histogram(fg_l.flatten(), bins=256, range=(0, 256), density=True)
            
            #Calculating the cumulative distribution function to balance contrast and brightness
            bg_cdf = bg_hist.cumsum()
            fg_cdf = fg_hist.cumsum()

            #Normalizing to 8-bit range
            bg_cdf = (bg_cdf / bg_cdf[-1]) * 255
            fg_cdf = (fg_cdf / fg_cdf[-1]) * 255
            
            #Linear interpolation to generate new pixel values
            bg_l_matched = np.interp(bg_l.flatten(), np.arange(0, 256), bg_cdf).reshape(bg_l.shape)
            
            #Join the channels back
            matched_lab = cv2.merge([bg_l_matched.astype(np.uint8), bg_a, bg_b])
            
            corrected = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
                
        elif mode == "color_transf":
            #Calculating the mean and standard deviation of each channel in the background image
            l_mean_bg, l_std_bg = bg_l.mean(), bg_l.std()
            a_mean_bg, a_std_bg = bg_a.mean(), bg_a.std()
            b_mean_bg, b_std_bg = bg_b.mean(), bg_b.std()
            
            l_mean_fg, l_std_fg = fg_l.mean(), fg_l.std()
            a_mean_fg, a_std_fg = fg_a.mean(), fg_a.std()
            b_mean_fg, b_std_fg = fg_b.mean(), fg_b.std()
            
            #Matching the mean and standard deviation 
            #It centers the values aroudn zero by subtracting the mean, scales them by multiplying over the ratio of the standard deviations, ensuring that the contrast levels match, 
            #and adding the foreground mean shifts the centered and scaled values in order to match the brightness
            l = (bg_l - l_mean_bg) * (l_std_fg / l_std_fg) + l_mean_fg
            a = (bg_a - a_mean_bg) * (a_std_fg / a_std_fg) + a_mean_fg
            b = (bg_b - b_mean_bg) * (b_std_fg / b_std_fg) + b_mean_fg
            
            #Clips odd values to the 8-bit range
            l = np.clip(l, 0, 255).astype(np.uint8)
            a = np.clip(a, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)
            
            chanel_merge = cv2.merge([l, a, b]) #Joins the channels again
            
            corrected = cv2.cvtColor(chanel_merge, cv2.COLOR_LAB2RGB) #Converts back to RGB
        else:
            raise ValueError(f'>> {mode} is not a valid mode: "histogram" or "color_transf".')
        
        return corrected
    
    
    
    @staticmethod
    def _smoothing(image, coef):
        """
        Slightly unorthodox way of both smoothing the picture and better matching the resolution of the background.
        This method degrades the quality of the image on purpose.
        
        Args:
            - image (numpy.ndarray): Image that requires smoothing.
            - coef (float): Smoothing coefficient.
        
        Returns:
            - smooth (numpy.ndarray): Smoothed and "degraded" image
        """
        if coef is not None:
            image_height, image_width, _ = image.shape

            new_height = int(image_height*coef)
            new_width = int(image_width*coef)
            
            #Bilinear interpolation so the image has less quality while still looking smooth
            image_resized = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_LINEAR) #Reduces the image size
            smooth = cv2.resize(image_resized, (image_width, image_height), interpolation = cv2.INTER_LINEAR) #Resizes back to the original size
            

            return smooth
        else:
            return image
        
        
        
    def advert_locations(self, image, conf_score = CONF_SCORE):
        """
        Method that processes the image and looks for all the possible places to insert the desired advert.
        It looks for the Region of Interest (RoI) bounding boxes and then performs Otsu thresholding for Edge Detection to
        predict rectangles more accurate of the edges and computes the transformation matrix that corrects
        the perspective.
        
        Args:
            - image (numpy.ndarray): Image where the advert will be placed.
        Params:
            - conf_score (float): Confidence score for the detection model.
        Returns:
            - cnt_list (list): Coordinates of all the unaltered bounding boxes.
            - warp_boxes (list): Coordinates of all the bounding boxes corrected for perspective.
        """

        image_height, image_width, _ = image.shape
        results = self.model.predict(image, imgsz = [image_height, image_width], conf = conf_score, verbose = False)
        
        cnt_list = [] #Stores the contours of the possible locations
        warp_boxes = [] #Bounding boxes warped for better perspective
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls)
                if class_id in [386, 527, 587, 588]: #Classes that we want to evaluate -> [Picture Frame, Television, Window, Window Blind]
                    box_coords = box.xyxy[0].tolist()
                    x_min, y_min, x_max, y_max = BRNDTS._box_margin(box_coords, 2)
                    
                    #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    #cv2.putText(image, name, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    roi = image[y_min:y_max, x_min:x_max] #Extracts only the Region of Interest (RoI) for efficiency
                    edges = BRNDTS._edge_detection(roi, 2.5, 0) #Binary image with tight edges
                    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Generates contours that approximate the edges
                    
                    #We assume that the best contour will be the outermost one, i.e, the one with bigger area
                    max_area = 0
                    best_contour = None
                    for contour in cnts:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            best_contour = contour
                            
                    if best_contour is not None:
                        rect=list(cv2.minAreaRect(best_contour)) #List of the center point, width, height and rotation angle of the best contour's rectangle
                        
                        if rect[2] == 0: #Removes unwanted rotations due to the approximation of the contour's rectangle
                            rect[2]=90
                            
                        rect_points = cv2.boxPoints(rect) #Corner points of the previous rectangle
                        rect_points = np.intp(rect_points)
                        rect_points = np.float32(rect_points)
                        
                        reorder_idx = [0, 1, 3, 2] #The coordinates need to be reorderd for future processing
                        reordered_rect_points = rect_points[reorder_idx]
                        
                        warp_boxes.append(reordered_rect_points) #Saving all the warped bounding boxes

                        x, y, w, h = cv2.boundingRect(best_contour)

                        placement_coords = [x, x_min, y, y_min, w, h]
                        cnt_list.append(placement_coords) #Saving all the unaltered boundig boxes


        return cnt_list, warp_boxes 
    

    
    def segment_occlusions(self, image, seg_conf_score = SEG_CONF_SCORE):
        """
        Method that improves realism by segmenting an occlusion and superimposing that extracted
        layer over the advert.
        
        Args:
            - image (numpy.ndarray): Image where the advert will be placed.
        Params:
            - seg_conf_score (float): Confidence score of the segmentation model.
        Returns:
            - occlusion_masks (numpy.ndarray): Composite binary mask of all the occlusions in the original image.
        """
        
        image_height, image_width, _ = image.shape
        seg_results = self.seg_model.predict(image, imgsz = [image_height, image_width], classes = [0, 74], retina_masks = True, conf = seg_conf_score, verbose = False) #List of all the occlusions in the picture
        
        
        binary_mask = np.zeros(image.shape[:2], np.uint8) #Prealocate memory to store the binary mask of each individual occlusion
        occlusion_masks = np.zeros((image_height, image_width, 4), dtype=np.uint8) #Prealocate memory to store the merged occlusions' masks

        for seg_r in seg_results:
            for c in seg_r:
                cnt = c.masks.xy.pop().astype(np.int32) #Contour of the occlusion
                cv2.drawContours(binary_mask, [cnt], -1, (255, 255, 255), cv2.FILLED) #Draws the contour onto a binary mask
                
                x, y, w, h = cv2.boundingRect(cnt) #Coordiantes of the binary mask rectangle

                roi = image[y:y+h, x:x+w] #Processing only the Region of Interest (RoI) relative to the occlusion to save memory
                
                roi_mask = binary_mask[y:y+h, x:x+w] #Binary mask of the RoI

                if roi.shape[2] == 3: #Checks if the RoI has an alpha channel (for transparency manipulation)
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

                alpha_channel = roi[:, :, 3] #Extracts the alpha channel

                # Bitwise "and" to retains the overlapping regions of the alpha channel and the roi
                new_alpha_channel = cv2.bitwise_and(alpha_channel, roi_mask)
                roi[:, :, 3] = new_alpha_channel #Modifies the old alpha channel with the new one
                

                alpha_s = roi[:, :, 3] / 255.0 #Normalizes the alpha channel of the source for blending
                alpha_l = 1.0 - alpha_s #Inverts the destination's alpha channel

                for color in range(3): #Blends the color channels
                    occlusion_masks[y:y+h, x:x+w, color] = (alpha_s * roi[:, :, color] + alpha_l * occlusion_masks[y:y+h, x:x+w, color])

                occlusion_masks[y:y+h, x:x+w, 3] = np.maximum(occlusion_masks[y:y+h, x:x+w, 3], roi[:, :, 3]) #Blends the alpha channels
        
        
        return occlusion_masks
    

    
    @staticmethod
    #There are a lot of arguments on purpose. Most of them should probably be hardcoded, however, I wanted some flexibility and decided to include them here to easily override the config.py file.
    def gen_advert(advert, image, cnt_list, seg_occlusion_masks, warp_boxes, down_scale = DOWN_SCALE, border_t = BORDER_T, border_color = BORDER_COLOR, warp = WARP, bordered = BORDERED, color_correction = COLOR_CORRECTION, smoothing = SMOOTHING, mode = MODE, save = SAVE):
        """
        Method that finalizes all the process and applies the advert to the background picture. It has the option to apply
        a correction to the perspective, add a border and choose where to place the ad.
        
        Args:
            - advert (numpy.ndarray): Advert iamge:
            - image (numpy.ndarray): Background image where the ad will be applied.
            - cnt_list (list): List of all the possible locations.
            - seg_occlusion_masks (numpy.ndarray): Mask of all the occlusions.
            - warp_boxes (list): List of all the possible locations corrected for perspective.
        Params:
            - down_scale (float): Reduces the size of the image to account for the border.
            - border_t (float): Border thickness.
            - border_color (tuple (R, G, B)): Color of the border.
            - warp (bool): If True, fix the perspective.
            - bordered (bool): If True, adds a border.
            - color_correction (str): Mode of color correction:
                            - histogram: Matches the histogram of the advertisement with the background image.
                            - color_transf: Adjusts the colors using the mean and standard deviation of the background image.
            - smoothing (float): Matches the ad apparent resolution to the source's and smoothes.
            - mode (str): Different mode of applications of the ad:
                            - area: Looks for the location with the bigger area
                            - random: Looks for a random location of the list of locations
                            - total: Applies the ad to all the locations in the list.
            - save (bool): If true, saves the picture to the current working directory.
        Returns:
            - image (numpy.ndarray): Final image with all the changes.
        """
        
        n_locations = len(cnt_list) #Total of the possible locations
        smoothing_str = f'{smoothing*100:.2f}%' if smoothing is not None else smoothing
            
        print(f'\n>> There are {n_locations} possible locations and the ad was placed with the following arguments: \
              \n  -> Mode: {mode} \
              \n  -> Down Scale: {down_scale*100:.2f}% \
              \n  -> Bordered: {bordered} \
              \n  -> Border Thickness: {border_t} \
              \n  -> Border Color: {border_color} \
              \n  -> Color Correction: {color_correction} \
              \n  -> Smoothing: {smoothing_str} \
              \n  -> Perspective Warp: {warp} \
              \n  -> Save: {save}')
        
        if color_correction is not None: 
            advert = BRNDTS._color_correction(advert, image, color_correction)
            
        #if smoothing is not None and bordered == False: #When the border needs to be sharp and the image smooth
        #    advert = BRNDTS._smoothing(advert, smoothing)
            
        if mode.lower() == 'total': #Mode that applies the advert to all the detected locations
            for i in range(0, n_locations):
                x, x_min, y, y_min, w, h = (value for value in cnt_list[i]) #Coordinates of the location's bounding box
                advert_resized = BRNDTS._resize_and_padding(advert, int(w*down_scale), int(h*down_scale)) #REsizes the advert to fit the bounding box and adds padding to empty space
                advert_height, advert_width, _ = advert_resized.shape

                if warp: #Applies the perspective warp
                    transform_matrix, dest_width, dest_height = BRNDTS._warp_advert(i, x, y, advert_height, advert_width, warp_boxes) #Applies the perspective correction
                    advert_warped = cv2.warpPerspective(advert_resized, transform_matrix, (dest_width, dest_height))

                    image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_warped #Applies the ad to the backgroudn image

                if bordered: #Applies a border to the advert for improved realism
                    advert_bordered = cv2.copyMakeBorder(advert_resized, 
                                                         int(advert_resized.shape[1]*border_t), 
                                                         int(advert_resized.shape[1]*border_t), 
                                                         int(advert_resized.shape[1]*border_t), 
                                                         int(advert_resized.shape[1]*border_t), 
                                                         cv2.BORDER_CONSTANT, 
                                                         value=border_color
                                                         )
                    
                    if smoothing is not None: #Needs to be here to also apply the smoothing to the border
                        advert_bordered = BRNDTS._smoothing(advert_bordered, smoothing)
                    
                    advert_height, advert_width, _ = advert_bordered.shape #Extracts new image dimension due to the addition of the border

                    image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_bordered #Applies the bordered ad to the image

                else: #Case where neither a warp or a border ar considered
                    image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_resized 

        else:

            if mode.lower() == 'random': #Mode that applies the advert to a random location of the possible list
                i = np.random.randint(0, n_locations)

            elif mode.lower() == 'area': #Mode that applies the advert to the location with the bigger area
                max_area = 0
                i = 0
                for idx in range(0, n_locations):
                    _, _, _, _, w, h = (value for value in cnt_list[idx]) 
                    area = w*h
                    if area > max_area:
                        max_area = area
                        i = idx

            #Starting here...
            x, x_min, y, y_min, w, h = (value for value in cnt_list[i]) 
            advert_resized = BRNDTS._resize_and_padding(advert, int(w*down_scale), int(h*down_scale))
            advert_height, advert_width, _ = advert_resized.shape

            if warp:
                transform_matrix, dest_width, dest_height = BRNDTS._warp_advert(i, x, y, advert_height, advert_width, warp_boxes) #Applies the perspective correction
                advert_warped = cv2.warpPerspective(advert_resized, transform_matrix, (dest_width, dest_height))
                image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_warped

            if bordered:
                advert_bordered = cv2.copyMakeBorder(advert_resized, 
                                                     int(advert_resized.shape[1]*border_t), 
                                                     int(advert_resized.shape[1]*border_t), 
                                                     int(advert_resized.shape[1]*border_t), 
                                                     int(advert_resized.shape[1]*border_t), 
                                                     cv2.BORDER_CONSTANT, 
                                                     value=border_color
                                                     )
                
                if smoothing is not None: #Needs to be here to also apply the smoothing to the border
                    advert_bordered = BRNDTS._smoothing(advert_bordered, smoothing)
                    
                advert_height, advert_width, _ = advert_bordered.shape
                
                image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_bordered

            else:
                image[y_min+y:y_min+y+advert_height, x_min+x:x_min+x+advert_width] = advert_resized    
            #... and ending here, it is the same as for the condition mode.lower() == 'total'
        

        #Performs the subtraction of the occlusions
        x, y = 0, 0 #The occlusion mask has the dimension of the original image, so the blending starts at the top left corner
        y_offset = seg_occlusion_masks.shape[0]
        x_offset = seg_occlusion_masks.shape[1]
        alpha_s = seg_occlusion_masks[:, :, 3] / 255.0 #Normalizes the source's alpha channel
        alpha_l = 1.0 - alpha_s #Inverts the destination's alpha channel
        
        #Blends all the color channels
        for c in range(0, 3):
            image[y:y+y_offset, x:x+x_offset, c] = (alpha_s * seg_occlusion_masks[:, :, c] + alpha_l * image[y:y+y_offset, x:x+x_offset, c])

        if save==True:
            cv2.imwrite(f'image_with_advert.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(f"\n>> Image saved in: {os.path.join(os.getcwd(), 'image_with_advert.jpg')}\n")

        return image      
            
        
        