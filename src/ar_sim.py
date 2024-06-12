from fastapi import FastAPI, Request,WebSocket,BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import uvicorn
import os
import sys
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
from pydantic import BaseModel  
from datetime import datetime
import cv2
from geojson_pydantic import Feature, FeatureCollection, Point, Polygon
from itertools import cycle
import time
import redis
import argparse
from shapely import wkt

from threading import Thread

class TreeProperties(BaseModel):
    x: float
    y: float
    z: float
    pixel_center: list
    diameter: float
    time: datetime

TreeResponse = Feature[Point,TreeProperties]

class PositionProperties(BaseModel):
    x: float
    y: float
    z: float
    azimuth: float
    pitch: float
    roll: float
    time: datetime

PositionResponse = Feature[Point,PositionProperties]

TreeResponseCollection = FeatureCollection[TreeResponse]


class DataHandler:
    def __init__(self,data_dir:str) -> None:

        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images/"
        rgb_pos_catalog_path = self.data_dir/'pos_rgb_catalog.csv'
        raw_tree_gjson_path = self.data_dir/'raw_trees.json'

        rgb_pos_catalog = pd.read_csv(rgb_pos_catalog_path)
        rgb_pos_catalog['geometry'] = rgb_pos_catalog['geometry'].apply(wkt.loads)
        rgb_pos_catalog['time'] = pd.to_datetime(rgb_pos_catalog['time'],format='mixed')
        self.rb_pos_catalog = gpd.GeoDataFrame(rgb_pos_catalog,geometry="geometry")
        with open(raw_tree_gjson_path,'r') as f:
            self.raw_tree_gjson = json.load(f)


        self.raw_tree_df = gpd.GeoDataFrame.from_features(self.raw_tree_gjson['features'])
        self.raw_tree_df['time'] = pd.to_datetime(self.raw_tree_df['time'],format='mixed')
        # get time deltas between each entry in the catalog
        time_diff = self.rb_pos_catalog['time'].diff().dt.total_seconds().to_list()
        time_diff[0] = 0
        self.time_delta = cycle(time_diff)

        self.time_cycle = cycle(self.rb_pos_catalog['time'].to_list())

        self.redis_server = redis.Redis(host='localhost',port=6379,db=0)
        self.tree_sub = self.redis_server.pubsub()
        self.pos_sub = self.redis_server.pubsub()
        self.tree_sub.subscribe('tree')
        self.pos_sub.subscribe('position')

    def iter(self,time:datetime):
        current_iter = self.rb_pos_catalog[self.rb_pos_catalog['time']==time].__geo_interface__["features"][0] 
        
        tolerance = pd.Timedelta(seconds=.5)
        tree_mask = (self.raw_tree_df['time'] >= time - tolerance) & (self.raw_tree_df['time'] <= time + tolerance)
        raw_tree_set = self.raw_tree_df.loc[tree_mask].__geo_interface__
        current_rgb = Path(current_iter["properties"]['rgb_path'])
        rgb_path = self.img_dir/current_rgb.name

        self.current_rgb_path = rgb_path
        
        self.current_iter = PositionResponse(**current_iter)
        
   
        self.redis_server.publish('position',self.current_iter.model_dump_json())

        if raw_tree_set:
            self.current_tree_set = TreeResponseCollection(**raw_tree_set)
            self.redis_server.publish('tree',self.current_tree_set.model_dump_json())

    def get_img(self):
        img = cv2.imread(str(self.current_rgb_path))
        ret,buffer = cv2.imencode('.jpg',img)
        frame = buffer.tobytes()
        if ret:
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def run(self):
        # Iterate over the time entries in self.rb_pos_catalog, when done restart the loop
        while True:
            for delta,time_stamp in zip(self.time_delta,self.time_cycle):
                self.iter(time_stamp)
                time.sleep(delta)

ar_sim = FastAPI()

@ar_sim.on_event("startup")
async def startup_event():
    
    ar_sim.datahandler = DataHandler("/home/dunbar/Desktop/Recordings/2024_5_10/ar_sim_test/")
    main_thread = Thread(target=ar_sim.datahandler.run)
    main_thread.start()

   

@ar_sim.get("/video")
async def video_feed():
    def stream():
        while True:
            for image in ar_sim.datahandler.get_img():
                yield image
    return StreamingResponse(stream(),media_type='multipart/x-mixed-replace; boundary=frame')

@ar_sim.get("/tree",)
async def tree_ws(request:Request) -> TreeResponseCollection:
 
    while True:
        for message in ar_sim.datahandler.tree_sub.listen():
            if message['type'] == 'message':
                response = TreeResponseCollection.model_validate_json(message['data'])
                return response
            
@ar_sim.websocket("/position")
async def pos_ws(websocket: WebSocket) -> PositionResponse:
    await websocket.accept()
    while True:
        for message in ar_sim.datahandler.pos_sub.listen():
            if message['type'] == 'message':
                response = PositionResponse.parse_raw(message['data'])
                await websocket.send_text(response)



if __name__ == '__main__':

  

    uvicorn.run("ar_sim:ar_sim",host="0.0.0.0", port=8800,log_level="info",reload=True)

            