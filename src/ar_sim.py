from fastapi import FastAPI, Request,WebSocket
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
    def __init__(self,img_dir,rgb_pos_catalog,raw_tree_gjson) -> None:
        self.img_dir = Path(img_dir)
        self.rb_pos_catalog = gpd.read_csv(rgb_pos_catalog)
        self.raw_tree_df = gpd.read_file(raw_tree_gjson)

        # get time deltas between each entry in the catalog
        time_diff = self.rb_pos_catalog['time'].diff().dt.total_seconds()
        self.time_delta = cycle(time_diff.to_list())

        self.time_cycle = cycle(self.rb_pos_catalog['time'].to_list())

        self.redis_server = redis.Redis(host='localhost',port=6379,db=0)
        self.tree_sub = self.redis_server.pubsub()
        self.pos_sub = self.redis_server.pubsub()
        self.tree_sub.subscribe('tree')
        self.pos_sub.subscribe('position')

    def iter(self,time:datetime):
        current_iter = self.rb_pos_catalog[self.rb_pos_catalog['time']==time][0]
        raw_tree_set = self.raw_tree_df[self.raw_tree_df['time']==time].__geo_interface__
        current_rgb = str(current_iter['rgb_path'].value())
        rgb_path = self.img_dir/current_rgb

        self.current_rgb_path = rgb_path
        self.current_tree_set = TreeResponseCollection(raw_tree_set.to_dict())
        self.current_iter = PositionResponse(current_iter.drop('rgb_path',axis=1).to_dict())
        
        self.redis_server.publish('tree',json.dumps(self.current_tree_set))
        self.redis_server.publish('position',json.dumps(self.current_iter))
    
    def get_img(self):
        img = cv2.imread(self.current_rgb_path)
        ret,buffer = cv2.imencode('.jpg',img)
        frame = buffer.tobytes()
        if ret:
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    def run(self):
        # Iterate over the time entries in self.rb_pos_catalog, when done restart the loop

        for delta,time in zip(self.time_delta,self.time_cycle):
            self.iter(time)
            time.sleep(delta)

ar_sim = FastAPI()

@ar_sim.on_event("startup")
async def startup_event():
    global datahandler
    ar_sim.datahandler = datahandler 
    ar_sim.datahandler.run()


@ar_sim.get("/video")
async def video_feed():
    return StreamingResponse(datahandler.get_img(),media_type='multipart/x-mixed-replace; boundary=frame')

@ar_sim.websocket("/tree",)
async def tree_ws(websocket: WebSocket) -> TreeResponseCollection:
    await websocket.accept()
    while True:
        for message in ar_sim.datahandler.tree_sub.listen():
            if message['type'] == 'message':
                response = TreeResponseCollection.parse_raw(message['data'])
                await websocket.send_text(response)
            
@ar_sim.websocket("/position")
async def pos_ws(websocket: WebSocket) -> PositionResponse:
    await websocket.accept()
    while True:
        for message in ar_sim.datahandler.pos_sub.listen():
            if message['type'] == 'message':
                response = PositionResponse.parse_raw(message['data'])
                await websocket.send_text(response)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,required=True)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    img_dir = data_dir/'images'
    rgb_pos_catalog = data_dir/'pos_rgb_catalog.csv'
    raw_tree_gjson = data_dir/'raw_trees.geojson'
    datahandler = DataHandler(img_dir,rgb_pos_catalog,raw_tree_gjson)
    uvicorn.run("ar_sim:ar_sim",host="0.0.0.0", port=8800,log_level="info",reload=True)

            