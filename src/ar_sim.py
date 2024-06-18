from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
from pathlib import Path
import pandas as pd
import geopandas as gpd
from pydantic import BaseModel,Field
from datetime import datetime
import cv2
from geojson_pydantic import Feature, FeatureCollection, Point
from itertools import cycle
import time
import redis
from shapely import wkt
from sse_starlette.sse import EventSourceResponse
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
import os
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional
from .stemmap import StemMap,SMPosition,NoTreesFound

class TreeProperties(BaseModel):
    x: float
    y: float
    z: float
    pixel_center: Optional[list]
    diameter: float
    time:datetime 

TreeResponse = Feature[Point,TreeProperties]

class PositionProperties(BaseModel):
    x: float
    y: float
    z: float
    azimuth: float
    pitch: float
    roll: float
    time:datetime 
    lat:float
    lon:float

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
        self.rb_pos_catalog["lat"] = self.rb_pos_catalog['geometry'].apply(lambda x: x.y)
        self.rb_pos_catalog["lon"] = self.rb_pos_catalog['geometry'].apply(lambda x: x.x)

        with open(raw_tree_gjson_path,'r') as f:
            self.raw_tree_gjson = json.load(f)


        self.raw_tree_df = gpd.GeoDataFrame.from_features(self.raw_tree_gjson['features'])
        self.raw_tree_df['time'] = pd.to_datetime(self.raw_tree_df['time'],format='mixed')
        # get time deltas between each entry in the catalog
        time_diff = self.rb_pos_catalog['time'].diff().dt.total_seconds().to_list()
        time_diff[0] = 0

        self.iter_count = cycle(range(len(time_diff)))
        self.time_delta = cycle(time_diff)

        self.time_cycle = cycle(self.rb_pos_catalog['time'].to_list())

        self.redis_server = redis.Redis(host='localhost', port=6379)
        self.tree_sub = self.redis_server.pubsub()
        self.pos_sub = self.redis_server.pubsub()
        self.global_tree_sub = self.redis_server.pubsub()
        self.global_tree_sub.subscribe('global_tree')
        self.tree_sub.subscribe('tree')
        self.pos_sub.subscribe('position')

        self.stem_map = StemMap(cam_info={"depth_distance":20,"fov_h":45})


    
    def event_process_trees(self):
        positions_processed = []
        while True:
            pos = self.redis_server.lpop('position')
            if pos is not None:
                pos = json.loads(pos.decode('utf-8'))

                input = pos['properties']
                input['elevation'] = pos['geometry']['coordinates'][-1]
                input['timestamp'] = input['time']
                positions_processed.append(SMPosition(**input))
            else:
                break
        
        try:
            new_trees = self.stem_map.event_process_staged_trees(positions=positions_processed)
            
            new_trees_gpd = gpd.GeoDataFrame(new_trees,geometry='geometry')
            new_trees_gpd.timestamp = new_trees_gpd.timestamp.apply(lambda x: x.isoformat())
    
            self.redis_server.publish('global_tree',new_trees_gpd.to_json())
        except NoTreesFound:
            pass

    def iter(self,time:datetime):
        current_iter = self.rb_pos_catalog[self.rb_pos_catalog['time']==time].__geo_interface__["features"][0] 
        
        tolerance = pd.Timedelta(seconds=.5)
        tree_mask = (self.raw_tree_df['time'] >= time - tolerance) & (self.raw_tree_df['time'] <= time + tolerance)
        raw_tree_set = self.raw_tree_df.loc[tree_mask]
        current_rgb = Path(current_iter["properties"]['rgb_path'])
        rgb_path = self.img_dir/current_rgb.name

        self.current_rgb_path = rgb_path
        
        self.current_iter = PositionResponse(**current_iter)
        

        self.redis_server.lpush('position',self.current_iter.model_dump_json())
        self.redis_server.publish('position',self.current_iter.model_dump_json())

        if raw_tree_set.shape[0]>0:
            to_stemmap = raw_tree_set.copy()
            to_stemmap["lat"] = to_stemmap.apply(lambda x: x.geometry.y,axis=1)
            to_stemmap["lon"] = to_stemmap.apply(lambda x: x.geometry.x,axis=1)
            to_stemmap['elev'] = to_stemmap.apply(lambda x: x.geometry.z,axis=1)
            to_stemmap["timestamp"] = to_stemmap["time"]
            self.stem_map.stage_tree(to_stemmap.to_dict(orient='records'))
            self.current_tree_set = TreeResponseCollection(**raw_tree_set.__geo_interface__)
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
            for delta,time_stamp,iter_count in zip(self.time_delta,self.time_cycle,self.iter_count):
                if iter_count == 0:
                    self.stem_map = StemMap(cam_info={"depth_distance":20,"fov_h":45})
                self.iter(time_stamp)
                time.sleep(delta)

ar_sim = FastAPI()
ar_sim.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@ar_sim.on_event("startup")
async def startup_event():
    
    data_dir = os.getenv("ARSIM_DATADIR")
    ar_sim.datahandler = DataHandler(data_dir)
    scheduler = BackgroundScheduler()
    scheduler.add_job(
            func=ar_sim.datahandler.event_process_trees,
            trigger="interval",
            minutes=2,
        )
    main_thread = Thread(target=ar_sim.datahandler.run)
    main_thread.start()
    scheduler.start()
   

@ar_sim.get("/video")
async def video_feed(request:Request):
    def stream():
        while True:
            for image in ar_sim.datahandler.get_img():
                yield image
    return StreamingResponse(stream(),media_type='multipart/x-mixed-replace; boundary=frame')

@ar_sim.get("/tree")
async def send_tree(request:Request) -> TreeResponseCollection:

    async def tree_event():
        while True:
            if await request.is_disconnected():
                break

            message = ar_sim.datahandler.tree_sub.get_message()
            if message:
                try:
                    response = TreeResponseCollection.model_validate_json(message['data'])
                    yield response.model_dump_json()
                except:
                    pass
    return EventSourceResponse(tree_event())

@ar_sim.get("/tree_global")
async def send_tree_filtered(request:Request) -> TreeResponseCollection:

    async def global_tree_event():
        while True:
            if await request.is_disconnected():
                break
            message = ar_sim.datahandler.global_tree_sub.get_message()
            if message:
                try:
                   
                    yield message['data']
                except:
                    pass
    return EventSourceResponse(global_tree_event())

@ar_sim.get("/position")
async def pos_sse(request:Request)-> PositionResponse:

    async def event_generator() :
        while True:
            if await request.is_disconnected():
                break

            message = ar_sim.datahandler.pos_sub.get_message()

            if message:
                try:
                    yield PositionResponse.model_validate_json(message['data']).model_dump_json()
                except:
                    pass

    return EventSourceResponse(event_generator())

