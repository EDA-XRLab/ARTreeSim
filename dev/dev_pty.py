import json
import os
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
from src.perceptree.detection_schema import Image,CameraModel,TreeProperties

if __name__ == "__main__":
    data_dir = dir / "ar_sim_test/images"
    images_jpg = [str(x) for x in list(data_dir.glob("*.jpg"))]
    images_jpg.sort(key=lambda x: int(os.path.split(x)[1][:-5]))
    redis_connection = redis.Redis(host='localhost', port=6379, db=0)
    redis_connection.flushall()
    # Download the data
    camera_model = CameraModel(c_col=320,c_row=240,f_col=500,f_row=500)

    count = 0
    if not (status := requests.get("http://localhost:8000/detections/start/")).ok:
        print("Error starting detection")
        print(status.text)

    print("Starting image stream")
    frame_time_s = 0.25
    for image in itertools.cycle(images_jpg):
        t1 = time.perf_counter()
        rgb = cv2.imread(str(image))
        depth = np.random.randint(2,20,size=(rgb.shape[0],rgb.shape[1],1)).astype(np.float64)
        depth += np.random.rand(*depth.shape)
        timestamp = datetime.datetime.now()
        image_proc = Image(rgb=rgb,depth=depth,model=camera_model,timestamp=timestamp)
        redis_connection.lpush("images",image_proc.model_dump_json())
        print('sent', image)
        while (det := redis_connection.lpop("detections")) is not None:
            det = TreeProperties(**json.loads(det))
            print(det)
            cv2.circle(rgb, (int(det.pixel_center[1]), int(det.pixel_center[0])), 2, (0, 0, 255), 2)
        
        cv2.imshow("test", rgb)
        t2 = time.perf_counter()
        wait_time = int((frame_time_s - (t2 - t1)) * 1000)
        print("DT: ", t2 - t1, wait_time)
        cv2.waitKey(max(wait_time, 1))
