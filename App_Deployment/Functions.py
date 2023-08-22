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


def Pipeline(image, mean_shape=(19, 19), diffr=0, div=0.4):
    '''Returns and appends a preprocessed image into a numpy array that the user creates for the train data'''

    #image = cv2.imread(image_path)
    # print(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (512, 512))

    # Smooth the image to remove noise
    smooth_image = cv2.blur(gray_image, (3, 3))

    # Calculate the mean and standard deviation of the pixel values using a 3x3 neighborhood
    mean_image = cv2.blur(smooth_image, mean_shape)
    std_image = cv2.absdiff(smooth_image, mean_image)

    # Define a threshold to determine crack regions
    # threshold = 10

    # Automated threshold selection based on histogram distribution in gray image, basically if there's so much brightness in
    # image we want to use a high threshold to activate the crack if not, we use a low threshold

    # Calculating the average of the 25th, 50th, 75th, and 85th percentile pixel value
    perc_25, perc_35, perc_45, perc_50, perc_65, perc_75 = np.percentile(gray_image, 25), np.percentile(gray_image, 35), \
                                                           np.percentile(gray_image, 45), np.percentile(gray_image,
                                                                                                        50), np.percentile(
        gray_image, 65), np.percentile(gray_image, 75)

    # Calculating average of the 1th, 1.2th and 1.5th percentiles
    perc_1, perc_1p2, perc_1p5 = np.percentile(gray_image, 1), np.percentile(gray_image, 1.2), \
                                 np.percentile(gray_image, 1.5)

    low_mean = np.mean([perc_1, perc_1p2, perc_1p5])
    high_mean = np.mean([perc_25, perc_35, perc_45, perc_50, perc_65, perc_75])
    join_mean = np.mean([perc_1, perc_1p2, perc_1p5, perc_25, perc_35, perc_45, perc_50, perc_65, perc_75])

    diff = high_mean - low_mean
    # ratio = low_mean/high_mean
    threshold = join_mean - low_mean

    # For non crack images, we don't expect a high difference so we set a condition
    if diff <= diffr:
        threshold = diffr
    else:
        threshold = np.sqrt(threshold) / div
    # print(threshold)

    # Create a binary mask where cracks are represented as white pixels
    crack_mask = np.zeros_like(std_image)
    crack_mask[std_image > threshold] = 255

    # Apply the crack mask to the original image
    gray_image = gray_image.astype(np.uint8)
    crack_mask = crack_mask.astype(np.uint8)
    segmented_image = cv2.bitwise_and(gray_image, gray_image, mask=crack_mask)

    # removing whiter pixels from the segmented image due to the Whitening effect
    segmented_image[segmented_image > join_mean + 15] = 0
    return segmented_image


