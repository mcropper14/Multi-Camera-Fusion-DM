import cv2
import numpy as np
import os
from load_data import load_matched_data

# Directory to store images
RAW_IMAGE_DIR = "raw_images_5_6"
UNDISTORTED_IMAGE_DIR = "undistorted_images_5_6"
WARPED_IMAGE_DIR = "warped_images_5_6"
STITCHED_IMAGE_DIR = "stitched_images_5_6_extrinsics2"

# Create directories
os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
os.makedirs(UNDISTORTED_IMAGE_DIR, exist_ok=True)
os.makedirs(WARPED_IMAGE_DIR, exist_ok=True)
os.makedirs(STITCHED_IMAGE_DIR, exist_ok=True)

# Camera 5 Intrinsics
K_5 = np.array([[657.473008030779, -0.412726794227289, 660.315113192361],
                [0, 659.828635018695, 513.577329202517],
                [0, 0, 1]])
dist_5 = np.array([-0.207272528619094, 0.0987355788097474, -0.0000970165500108280, 0.000475400200128317, -0.0230034042388187])

# Camera 6 Intrinsics
K_6 = np.array([[662.42612, 0, 645.69177],
                [0, 663.45887, 519.75771],
                [0, 0, 1]])
dist_6 = np.array([-0.17225, 0.04636, 0.00025, -0.00145])

# Extrinsic Transformation (T_05CAMERA_06CAMERA)
T_05CAMERA_06CAMERA = np.array([
    [0.591998047079647, 0.00845869872400891, 0.805895007224750, 0.395397757803753],
    [-0.0105281263693131, 0.999940764279252, -0.00276160963512663, 0.00415802375491464],
    [-0.805870629077047, -0.00684969696566936, 0.592052033897734, -0.255930509994327],
    [0.0, 0.0, 0.0, 1.0]
])

def save_image(image, path):
    """Saves the image to a given path."""
    cv2.imwrite(path, image)

def undistort_image(image, K, dist_coeffs):
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(image, K, dist_coeffs, None, new_K)
    x, y, w, h = roi
    return undistorted_img[y:y+h, x:x+w]

def warp_camera_5_to_6(img5):
    h, w = img5.shape[:2]
    R = T_05CAMERA_06CAMERA[:3, :3]
    t = T_05CAMERA_06CAMERA[:3, 3]
    # Including intrinsic matrix K_6 and converting transformation to homography
    H = K_6 @ np.hstack((R[:, :2], t.reshape(-1, 1)))

    # Normalize homography to maintain scale consistency
    H /= H[2, 2]

    img5_warped = cv2.warpPerspective(img5, H, (w, h))
    return img5_warped

def process_stitching(image_path_cam5, image_path_cam6, timestamp):
    img5 = cv2.imread(image_path_cam5)
    img6 = cv2.imread(image_path_cam6)

    if img5 is None or img6 is None:
        print(f"[ERROR] Missing image for {image_path_cam5} or {image_path_cam6}")
        return

    save_image(img5, f"{RAW_IMAGE_DIR}/raw_5_{timestamp}.jpg")
    save_image(img6, f"{RAW_IMAGE_DIR}/raw_6_{timestamp}.jpg")

    img5_undistorted = undistort_image(img5, K_5, dist_5)
    img6_undistorted = undistort_image(img6, K_6, dist_6)

    save_image(img5_undistorted, f"{UNDISTORTED_IMAGE_DIR}/undistorted_5_{timestamp}.jpg")
    save_image(img6_undistorted, f"{UNDISTORTED_IMAGE_DIR}/undistorted_6_{timestamp}.jpg")

    img5_warped = warp_camera_5_to_6(img5_undistorted)

    save_image(img5_warped, f"{WARPED_IMAGE_DIR}/warped_5_{timestamp}.jpg")


# Load matched GNSS-camera data
matched_data = load_matched_data()

for entry in matched_data:
    timestamp = entry["timestamp"]
    image_paths = entry["images"]

    cam5_path = image_paths.get("image_05")
    cam6_path = image_paths.get("image_06")

    if cam5_path and cam6_path:
        process_stitching(cam5_path, cam6_path, int(timestamp))
    else:
        print(f"[WARNING] Missing images for timestamp {timestamp}, skipping.")

print("\nProcess Completed!")
