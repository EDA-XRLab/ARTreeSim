#!/venv/bin/python3
from fastapi import FastAPI, Request,BackgroundTasks
from fastapi.responses import StreamingResponse,FileResponse,JSONResponse
import json
import pandas as pd
from datetime import datetime
import time
import redis
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
import os

from typing import Optional, List, Dict, Any,Union
import redis
from threading import Thread
from enum import Enum
import uvicorn
import sys
sys.path.append(".")
from perceptree import Perceptree
from detection_schema import Image,TreeProperties 

class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
   
def remove_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def run_detection_thread(app_object):
    print("Running detection")
    while app_object.is_running:
        
        try:
            detection_image = app_object.redis.lpop("images")
            if detection_image:
                detection_image = json.loads(detection_image)
                detection_image_proc = Image.from_serial(detection_image)
                detection_image_proc.model = dict(detection_image_proc.model)
                raw_detections = app_object.perceptree.predict(**dict(detection_image_proc))
                detections = [TreeProperties(**x) for x in raw_detections]
                count = len(detections) + app_object.system_status.get("detection_count",0)
                app_object.system_status["detection_count"] = count
                [app_object.redis.lpush("detections",x.model_dump_json()) for x in detections]
        except KeyboardInterrupt:
            app_object.system_status["end_time"] = datetime.now().isoformat()
            app_object.is_running = False
            break

detection_app = FastAPI()
detection_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@detection_app.on_event("startup")
def startup_event():
 
    detection_app.perceptree = Perceptree()
    detection_app.redis = redis.Redis(host='localhost', port=6379, db=0)
    detection_app.redis.flushall()
    detection_app.is_running = False

@detection_app.get("/detections/status")
async def get_status():

    if not detection_app.is_running:
        return JSONResponse({"error":"Detection not running"},status_code=400)
    return detection_app.system_status

@detection_app.get("/detections/start/")
async def run_detection(request: Request):
    if detection_app.is_running:
        return JSONResponse("Detection already running",status_code=200)
    
    detection_app.is_running = True
    start_time = str(datetime.now().isoformat())
    detection_app.system_status = {"start_time":start_time}

    detection_app.detection_thread = Thread(target=run_detection_thread,args=(detection_app,))
    detection_app.detection_thread.start()

    return JSONResponse(f"Detection started at {start_time}",status_code=200)

@detection_app.get("/detections/stop/")
async def stop_detection(request: Request):
    if not detection_app.is_running:
        return JSONResponse({"error":"Detection not running"},status_code=200)
    detection_app.system_status["end_time"] = datetime.now().isoformat()
    detection_app.is_running = False
    detection_app.detection_thread.join()
    time.sleep(5)

    return detection_app.system_status

@detection_app.post("/images/add")
async def add_image(image: Image):
    detection_app.redis.lpush("images",image.model_dump())
    return {"status":"Image added to detection queue"}

@detection_app.get("/detections/get")
async def get_detections(format: Optional[OutputFormat] = OutputFormat.JSON,
                         clear: Optional[bool] = False
                         ) -> Union[List[TreeProperties],Any]:

    try:
        detections_raw = [json.loads(x) for x in detection_app.redis.lrange("detections",0,-1)]
    except Exception as e:
        response = JSONResponse({"error":str(e)},status_code=500)
        return response
    
    detections = [TreeProperties(**x) for x in detections_raw]
    if not detections:
        return JSONResponse({"error":"No detections available"})
    if clear:
        detection_app.redis.flushall()
    background_tasks = BackgroundTasks()

    if format == OutputFormat.JSON:
        return detections
    
    elif format == OutputFormat.CSV:
        df = pd.DataFrame([x.model_dump() for x in detections])
        file_path = f"{detection_app.system_status['start_time']}_detections.csv"
        df.to_csv(file_path)
        background_tasks.add_task(remove_file,file_path)
        return FileResponse(file_path,background=background_tasks,filename=file_path,media_type="text/csv")

    else:
        return JSONResponse({"error":"Invalid output format"})
    


if __name__ == "__main__":
    uvicorn.run(detection_app,host="localhost",port=8000)