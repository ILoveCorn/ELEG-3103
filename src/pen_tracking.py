import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import heapq
from matplotlib import pyplot as plt

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
# Set the depth sensor to 'short range' mode
depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)
depth_scale = depth_sensor.get_depth_scale()

# Getting the depth camera's intrinsics
intrinsic = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.5
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Trajectory of the pen
traj = np.zeros((720, 1280, 3), dtype=np.uint8)

# Streaming loop
try:
    while True:
        # Start timing
        e1 = cv.getTickCount()

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image_bgr = np.asanyarray(color_frame.get_data())

        # define range of blue color in HSV
        lower_blue = np.array([100, 100, 60])
        upper_blue = np.array([120, 255, 255])

        # Threshold the HSV image to get only blue colors
        color_image_hsv = cv.cvtColor(color_image_bgr, cv.COLOR_BGR2HSV)
        mask_blue = cv.inRange(color_image_hsv, lower_blue, upper_blue)
        mask_blue_3d = np.dstack((mask_blue, mask_blue, mask_blue))

        # Bitwise-AND mask_blue and original image
        color_image_blue = cv.bitwise_and(color_image_bgr, color_image_bgr, mask=mask_blue)

        # Remove background - Set pixels further than clipping_distance to black
        bg_color = 0
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels

        fg = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
                      bg_color,
                      color_image_bgr)

        fg_gray = cv.cvtColor(fg, cv.COLOR_BGR2GRAY)
        ret, fg_bin = cv.threshold(fg_gray, 0, 255, cv.THRESH_BINARY)

        # Remove non-blue pixels in foreground
        markers = np.where(mask_blue_3d < 255, bg_color, fg)

        # Find contours of markers
        ret, thresh = cv.threshold(markers[:, :, 0], 1, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        markers_with_contours = markers.copy()
        cv.drawContours(markers_with_contours, contours, -1, (0, 255, 0), 1)

        # Draw bounding rectangles of markers
        markers_with_box = markers.copy()
        markers_num = 0
        x_pix, y_pix, depth = [], [], []
        for cnt in contours:
            # Check if current cnt belongs to a marker
            area = cv.contourArea(cnt)
            if area < 20:
                continue
            # Find and draw bounding rectangles
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(color_image_bgr, [box], 0, (0, 255, 0), 2)
            # Update list
            u = round(float(rect[0][0]))
            v = round(float(rect[0][1]))
            x_pix.append(u)
            y_pix.append(v)
            depth.append(depth_image.item((v, u)) * depth_scale)
            markers_num += 1

        # Convert (x_pix, y_pix, depth) to (x_3d_cam, y_3d_cam, z_3d_cam)
        x_3d_cam, y_3d_cam, z_3d_cam = [], [], []
        for u, v, d in zip(x_pix, y_pix, depth):
            pos_3d_cam = rs.rs2_deproject_pixel_to_point(intrinsic, (u, v), d)
            x_3d_cam.append(pos_3d_cam[0])
            y_3d_cam.append(pos_3d_cam[1])
            z_3d_cam.append(pos_3d_cam[2])
            # Record point to trajectory
            if d > 0.23:
                cv.circle(traj, [u, v], 3, [0, 0, 255], -1)

        if markers_num > 0:
            # Print marker position
            print('x_3d_cam:%s' % str(x_3d_cam))
            print('y_3d_cam:%s' % str(y_3d_cam))
            print('z_3d_cam:%s \n\n' % str(z_3d_cam))

            # Display markers' position
            font = cv.FONT_HERSHEY_SIMPLEX
            index = 0
            for u, v, x, y, z in zip(x_pix, y_pix, x_3d_cam, y_3d_cam, z_3d_cam):
                pos_str = 'P%d:(%.2f,%.2f,%.2f)' % (index, 100 * x, 100 * y, 100 * z)  # Display marker position in cm
                cv.putText(color_image_bgr, pos_str, (u + 15, v + 15), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                index += 1

        # End timing
        e2 = cv.getTickCount()
        fps = cv.getTickFrequency() / (e2 - e1)

        # Display fps
        font = cv.FONT_HERSHEY_SIMPLEX
        fps_str = 'fps:%.1f' % fps
        cv.putText(color_image_bgr, fps_str, (10, 20), font, 0.6, (0, 255, 0), 1, cv.LINE_AA)

        # Show tracking result
        cv.namedWindow('result', cv.WINDOW_NORMAL)
        cv.imshow('result', np.concatenate((color_image_bgr, traj), 0))

        # Key monitoring
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            break
finally:
    pipeline.stop()
