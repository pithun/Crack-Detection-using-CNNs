import cv2
import numpy as np

def extract_frames(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    # print(fps)

    # Initialize frame counter and time
    frame_count = 0
    time = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()
        # print(ret)

        # If frame was not successfully read, end the loop
        if not ret:
            break

        # Calculate the time in seconds
        time = frame_count / fps
        # print(int(time))

        # Check if the current frame's time is a multiple of 1 second
        if int(time) % 1 == 0:
            # Write the frame to an image file
            frame_output_path = f"{output_path}/frame_{int(time)}.jpg"
            cv2.imwrite(frame_output_path, frame)

        # Increment the frame counter
        frame_count += 10
        # print('Frame')

    # Release the video file
    video.release()

#def frame_to_video(folder_path):
    # Sorting the frames in

def apply_area_thresholding(crack_mask, Tarea=10):
    # Find connected components and their stats (including area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crack_mask, connectivity=8)

    # Define a minimum area threshold to remove small objects (adjust this value as needed)
    min_area_threshold = Tarea  # Adjust this threshold as needed

    # Create a mask to store the selected components
    selected_components = np.zeros_like(crack_mask)

    # Iterate through connected components
    for label in range(1, num_labels):
        # Check the area of the current component
        area = stats[label, cv2.CC_STAT_AREA]
        #print(area)

        # If the area is greater than or equal to the threshold, include it in the selected components
        if area >= min_area_threshold:
            selected_components[labels == label] = 255  # Set pixels to 255 (white)

    return selected_components

def find_endpoints(thinned_image):
    endpoints = []
    height, width = thinned_image.shape

    # We loop from (1,1) so that each pixel will have eight neighbours (0,0) is the edge
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            if thinned_image[x, y] == 1:  # Check if it's a white pixel (part of skeleton)
                # Define a neighborhood of 8 pixels around the current pixel
                neighborhood = thinned_image[x-1:x+2, y-1:y+2]

                # Count the number of non-zero (white) pixels in the neighborhood
                white_pixel_count = np.count_nonzero(neighborhood)
                #print(white_pixel_count)

                # If there's only one white pixel in the neighborhood, it's an endpoint
                if white_pixel_count == 2:#
                    endpoints.append((x, y))
    return endpoints

# Step 5: Length Threshold
def remove_small_skeletons(thinned_image, Tlength): 
    _, labels = cv2.connectedComponents(thinned_image)
    
    # The number of labels is max label + 1 since we do 0 indexing
    num_labels = labels.max() + 1
    
    # We create a list to store the skeletons or say the skeleton images where the >= Tlength condition is satisfied
    skeletons = []

    for label in range(1, num_labels):  # Exclude background (label 0)
        # We loop through all the labels and the variable "component" just creates an image where only pixels from a 
        # Particular label is shown
        component = (labels == label).astype(np.uint8)
        #print(component.shape)
        
        
        # Below after getting all pixels from each label, we sum the pixels from the labels and use basically if the sum 
        # of the skeleton from a particular label is small, we take it as noise which makes a whole lot of sense.
        if np.sum(component) >= Tlength:
            skeletons.append(component)

    # Create a new thinned image with the remaining skeletons - we basically join all the skeletons in the skeleton list
    # The skeleton list is where we kept the pixels with long lengths
    thinned_image = np.zeros_like(thinned_image)
    for skeleton in skeletons:
        thinned_image |= skeleton  # Use logical OR to combine skeletons

    return thinned_image

def implement_radius_restoration(tlength_applied_thin_img, original_image, Tradius= 20):   
    endpoints = find_endpoints(tlength_applied_thin_img)
    #print(endpoints)

    for endpoint in endpoints:
        x, y = endpoint[1], endpoint[0]
        mask = np.zeros_like(original_image)
        cv2.circle(mask, (x, y), Tradius, 255, -1)
        #plt.imshow(mask, cmap='gray')
        #plt.show()
    # Restore pixels in the thinned image using the mask
        tlength_applied_thin_img = np.maximum(tlength_applied_thin_img, cv2.bitwise_and(original_image, mask)) 
        
    return tlength_applied_thin_img

# Basically, this part is to restore the skeletons based after using tlength to remove stuff. The idea is to use dilation with
# a kernel which tells how much pixels to add basically to grow back pixels in the skeleton from the original image. 
# We then use bitwise and which returns 1 if both pictures are 1 and 1. What we're doing is that with dilation, 
# we're growing pixels but we want to keep only pixels based on their presence in out dirty image.
def restore_skeletons(original_image, thinned_image, radius = 1):   
    # Dilation process
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    
    # Here, we're basically doing dilation on the thinned image after that, we do the bitwise and on the dilated result and mask
    # and update the variable "expanded", thinned image on first iteration is still the skeleton but then we check if the 
    # thinned image == expanded(i.e dilated+mask) we then update thinned to be = expanded. We do a dilation on the new updated
    # thinned which takes the variable expanded, we then use bitwise_and on dilation 2 and original image and update expanded.
    # Now, thinned is first comparison of dilation and mask the idea is we want to stop when the result of a previous 
    # dilation and mask operation if it's the same as the current dilation and mask bitwise operation
    while True:
        expanded = cv2.dilate(src=thinned_image, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=original_image, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (thinned_image == expanded).all():
            return expanded
        thinned_image = expanded
        

def Pipeline (image, kernel = (31,31), div=0.4, Tarea = 12, Tlength = 43, Tradius = 18):
    # Convert the images to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Smooth the image to remove noise
    smooth_image = cv2.blur(gray_image, (3, 3))
    
    # Resizing both images
    gray_image = cv2.resize(smooth_image, (400, 400))

    # applying contrast and brightness enhancement
    alpha = 1 # Simple contrast control
    beta = 50   # Simple brightness control   
    gray_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

    # Calculate the mean and standard deviation of the pixel values using a 3x3 neighborhood
    mean_image = cv2.blur(gray_image, kernel)

    gray_image = gray_image.astype(np.int32)
    mean_image = mean_image.astype(np.int32)

    std_image = gray_image - mean_image

    # Automated threshold selection based on histogram distribution in gray image, basically if there's so much brightness in
    # image we want to use a high threshold to activate the crack if not, we use a low threshold

    # Calculating the average of the 25th, 50th, 75th, and 85th percentile pixel value
    perc_25, perc_35, perc_45, perc_50, perc_65, perc_75  = np.percentile(smooth_image, 25), np.percentile(smooth_image, 35), \
    np.percentile(smooth_image, 45), np.percentile(smooth_image, 50), np.percentile(smooth_image, 65), np.percentile(smooth_image, 75)

    # Calculating average of the 1th, 1.2th and 1.5th percentiles
    perc_1, perc_1p2, perc_1p5 = np.percentile(smooth_image, 1), np.percentile(smooth_image, 1.2), \
    np.percentile(smooth_image, 1.5)

    low_mean = np.mean([perc_1, perc_1p2, perc_1p5])
    high_mean = np.mean([perc_25, perc_35, perc_45, perc_50, perc_65, perc_75])
    join_mean = np.mean([perc_1, perc_1p2, perc_1p5, perc_25, perc_35, perc_45, perc_50, perc_65, perc_75])
    
    diff = high_mean - low_mean
    threshold = join_mean - low_mean
    #ratio = low_mean/high_mean
    
    threshold = (np.sqrt(threshold)/div)*-1
    # print(threshold)
    #print('Your threshold is {}'.format(threshold))
    
    # Create a binary mask where cracks are represented as white pixels
    crack_mask = np.zeros_like(std_image)
    crack_mask[std_image <= threshold] = 255

    # Apply the crack mask to the original image
    gray_image = gray_image.astype(np.uint8)
    crack_mask = crack_mask.astype(np.uint8)
    #segmented_image = cv2.bitwise_and(gray_image, gray_image, mask=crack_mask)
    
    # Applying Skele-Marker
    # 1. Area thresholding on crack image
    sk_crack_mask = apply_area_thresholding(crack_mask, Tarea)
    
    # 2. Thinning and using Tlength
    thinned_image = cv2.ximgproc.thinning(sk_crack_mask)
    
    # 3. Remove small skeletons
    thinned_image = remove_small_skeletons(thinned_image, Tlength)

    # 4. Restore skeletons
    restored_image = restore_skeletons(sk_crack_mask, thinned_image)

    # 5. Using Radius restoration
    rad = implement_radius_restoration(thinned_image, sk_crack_mask, Tradius = Tradius)
    
    # Combining remaining skeleton with radius based on endpoint restoration
    sk_crack_mask = cv2.bitwise_or(restored_image*255, rad)
    
    return sk_crack_mask
