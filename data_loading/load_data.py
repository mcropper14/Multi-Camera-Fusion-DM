import os
import cv2
import pandas as pd
import glob
from datetime import datetime

DATASET_DIR = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002"  


# Define camera folders
CAMERAS = ["image_00", "image_01", "image_02", "image_03", "image_04", "image_05", "image_06"]

def load_camera_timestamps():

    camera_timestamps = {}
    camera_image_paths = {}

    for cam in CAMERAS:
        timestamp_file = os.path.join(DATASET_DIR, "labeled", cam, "timestamps.txt")
        timestamps = pd.read_csv(timestamp_file,  header=None, names=["timestamp"])
        
        timestamps["timestamp"] = timestamps["timestamp"].apply(
            lambda x: datetime.strptime(x[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        )
        
        camera_timestamps[cam] = timestamps["timestamp"].tolist()

    # Find common timestamps across all cameras
    common_timestamps = set(camera_timestamps[CAMERAS[0]])  
    for cam in CAMERAS[1:]:
        common_timestamps.intersection_update(set(camera_timestamps[cam]))  
    
    common_timestamps = sorted(common_timestamps)

    #assign image to timestamp
    for i, timestamp in enumerate(common_timestamps):
        image_paths = {
            cam: os.path.join(DATASET_DIR, "labeled", cam, "data", f"{str(i).zfill(10)}.png")
            for cam in CAMERAS
        }
        camera_image_paths[timestamp] = image_paths  # Store paths in dictionary

    return common_timestamps, camera_image_paths

#mtach gns with time
def load_gnss_ins_data():
    """
    Reads and matches GNSS/INS data files with camera timestamps.
    Returns:
        - gnss_data_dict: Dictionary where keys are timestamps and values are GNSS/INS data.
    """
    ah_fuck = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel"
    gnss_dir = "data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel/data" 
    gnss_files = sorted(glob.glob(os.path.join(gnss_dir, "*.txt")))  # Get all GNSS text files
    

    #/home/cropthecoder/Documents/cadc_devkit/data/data/cadcd/2018_03_06/data/cadcd/2018_03_06/0001/data/cadcd/2018_03_06/0002/labeled/novatel/timestamps.txt

    gnss_timestamps_file = os.path.join(ah_fuck, "timestamps.txt")
    gnss_timestamps = pd.read_csv(gnss_timestamps_file, header=None, names=["timestamp"])
    
    gnss_timestamps["timestamp"] = gnss_timestamps["timestamp"].apply(
        lambda x: datetime.strptime(x[:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    )
    gnss_timestamps_list = gnss_timestamps["timestamp"].tolist()
    
    
    column_names = [
        "latitude", "longitude", "altitude", "undulation",
        "latitude_std", "longitude_std", "altitude_std",
        "roll", "pitch", "azimuth",
        "roll_std", "pitch_std", "azimuth_std",
        "ins_status", "position_type"
    ]
    
    gnss_data_dict = {}

    for i, gnss_file in enumerate(gnss_files):
        gnss_data = pd.read_csv(gnss_file, sep=r"\s+",  names=column_names)
        gnss_timestamp = gnss_timestamps_list[i]
        gnss_data_dict[gnss_timestamp] = gnss_data.iloc[0].to_dict()  # Store row as dict
    
    return gnss_data_dict

def match_gnss_to_camera(matched_timestamps, gnss_data_dict):

    matched_gnss_data = {}
    gnss_timestamps_sorted = sorted(gnss_data_dict.keys())

    for cam_timestamp in matched_timestamps:
        closest_gnss_timestamp = min(gnss_timestamps_sorted, key=lambda x: abs(x - cam_timestamp))
        matched_gnss_data[cam_timestamp] = gnss_data_dict[closest_gnss_timestamp]

    return matched_gnss_data


def load_matched_data():

    matched_timestamps, camera_image_paths = load_camera_timestamps()
    gnss_data_dict = load_gnss_ins_data()
    matched_gnss_data = match_gnss_to_camera(matched_timestamps, gnss_data_dict)

    matched_data = []
    for timestamp in matched_timestamps:
        if timestamp in matched_gnss_data:
            matched_data.append({
                "timestamp": timestamp,
                "images": camera_image_paths[timestamp],
                "gnss": matched_gnss_data[timestamp]
            })
    
    return matched_data

matched_data = load_matched_data()

print("\nExample Matched Entry:")
print("Timestamp:", matched_data[0]["timestamp"])
print("Images:", matched_data[0]["images"])
print("GNSS Data:", matched_data[0]["gnss"])


