from pathlib import Path
import redis
import time
import requests
import itertools
import numpy as np
import cv2
import datetime
import sys

dir = Path(__file__).parent.parent
sys.path.append(str(dir))
from src.perceptree.detection_schema import Image,CameraModel

if __name__ == "__main__":
    data_dir = Path.home()/"Desktop/Recordings/2024_5_10/ar_sim_test/images/"
    images_jpg = [str(x) for x in list(data_dir.glob("*.jpg"))]
    redis_connection = redis.Redis(host='localhost', port=6379, db=0)
    redis_connection.flushall()
    # Download the data
    camera_model = CameraModel(c_col=320,c_row=240,f_col=500,f_row=500)

    count = 0
    if not (status := requests.get("http://localhost:8000/detections/start/")).ok:
        print("Error starting detection")
        print(status.text)
    print("Starting image stream")
    for image in itertools.cycle(images_jpg):
        rgb = cv2.imread(str(image))
        depth = np.random.randint(2,20,size=(rgb.shape[0],rgb.shape[1],1)).astype(np.float64)
        depth += np.random.rand(*depth.shape)
        timestamp = datetime.datetime.now()
        image_proc = Image(rgb=rgb,depth=depth,model=camera_model,timestamp=timestamp)
       
        redis_connection.lpush("images",image_proc.model_dump_json())
        time.sleep(1)
