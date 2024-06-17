from typing import List,Optional,Tuple,Union
import logging
import uuid
import pandas as pd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point,Polygon
from shapely import unary_union, transform
from shapely.affinity import rotate
import numpy as np
import geopandas as gpd
import datetime
import pymap3d as pm
from copy import deepcopy
from functools import partial
from pydantic import BaseModel

logger = logging.getLogger(__file__)

class SMPosition(BaseModel):
    x: float
    y: float
    z: float
    timestamp: Optional[datetime.datetime]
    lat: Optional[float] = 0
    lon: Optional[float] = 0
    elevation: Optional[float] = 0
    azimuth: Optional[float] = 0

    class Config:
        extra = 'ignore'

class NoTreesFound(Exception):
    def __init__(self):
        self.message = "No staged trees to process"
        super().__init__(self.message)


class StemMap:

    pre_processing_dbscan_args = {
        "eps": 1,
        "min_samples": 2,
        "dbh_max": 1.5,
        "dbh_min": 0.05,
    }
    post_processing_dbscan_args = {
        "eps": 1,
        "min_samples": 1,
    }

    def __init__(self, cam_info: dict, tree_df: Optional[pd.DataFrame] = None):
        # Need the camera's field of view and depth distance

        self.cam_info = cam_info
        if tree_df is None:
            self.tree_df = pd.DataFrame()
        else:
            self.tree_df = tree_df
       

        self.tree_df = self.tree_df.assign(
            x=lambda df: df.get('x', pd.Series()),
            y=lambda df: df.get('y', pd.Series()),
            z=lambda df: df.get('z', pd.Series()),
            diameter=lambda df: df.get('diameter', pd.Series()),
            timestamp=lambda df: df.get('timestamp', pd.Series()),
            id=lambda df: df.get('id', pd.Series()),
            status=lambda df: df.get('status', pd.Series()),
            harvest_timestamp=lambda df: df.get('harvest_timestamp', pd.Series()),
            geometry= lambda df: df.get('geometry',pd.Series())
        )
        self.camera_viewshed = self.generate_viewshed_polygon(
            self.cam_info["depth_distance"], self.cam_info["fov_h"]
        )

        self.staged_predictions = []

        # TODO Add data validation for the tree_df using pandera?

    def add_trees(
        self, trees: List[dict], positions: List[SMPosition]
    ) -> Union[pd.DataFrame, None]:
        """
        Add trees to the stemmap and update the stemmap with aligned trees.

        Args:
            trees (List[dict]): List of tree predictions.
            positions (List[Position]): List of positions.

        Returns:
            Union[pd.DataFrame, None]: DataFrame containing the newly added trees, or None if no new trees were added.
        """
        processed_trees: pd.DataFrame = self.process_and_cluster_predictions(
            trees, **self.pre_processing_dbscan_args
        )
        processed_trees["geometry"] = processed_trees.apply(lambda x: Point(x.lon,x.lat),axis=1)
        log = f"Distilled {len(processed_trees)} trees from {len(trees)} predictions"
        logger.info(log)

        processed_trees["count"] = 1
        processed_trees["id"] = [uuid.uuid4().hex for _ in range(len(processed_trees))]

        viewshed_map: Polygon = self.generate_viewshed_map(positions)
        # Query trees from the current stem map that are within the viewshed
        viewshed_map_bounds: Tuple = viewshed_map.bounds
        viewshed_query_log = (
            f"Querying stemmap for trees within viewshed bounds: {viewshed_map_bounds}"
        )
        logger.info(viewshed_query_log)
        trees_in_viewshed = self.tree_df[
            self.tree_df.apply(
                lambda x: viewshed_map.contains(Point(x["lon"], x["lat"])), axis=1
            )
        ]
        found_trees_log = (
            f"Found {trees_in_viewshed.shape[0]} trees within the viewshed"
        )
        logger.info(found_trees_log)

        if trees_in_viewshed.shape[0] > 0:

            # Align the trees in the viewshed to the new stemmap
            aligned_trees, alignment_info = self.align_sub_stemmaps(
                trees_in_viewshed, processed_trees
            )

            current_timestamp = max(processed_trees["timestamp"])
            harvest_log = f"Identified {alignment_info['harvest_count']} trees as harvested at {current_timestamp}"
            logger.info(harvest_log)
            update_log = f"Updated {alignment_info['update_count']} trees in the stemmap"
            logger.info(update_log)

            new_trees = aligned_trees[aligned_trees["status"] == "new"].copy()
            self.gen_new_tree_log(new_trees)

            intermediate = pd.concat([aligned_trees,self.tree_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)
            intermediate.loc[intermediate.status.isna(),"status"] = "old"
            self.tree_df = intermediate


            current_timestamp = max(processed_trees["timestamp"])
            harvest_log = f"Identified {alignment_info['harvest_count']} trees as harvested at {current_timestamp}"
            logger.info(harvest_log)
            update_log = f"Updated {alignment_info['update_count']} trees in the stemmap"
            logger.info(update_log)

            if new_trees.shape[0] > 0:
                return new_trees
        else:
            self.gen_new_tree_log(processed_trees)
    
            self.tree_df = pd.concat([processed_trees, self.tree_df])
            return processed_trees



    def gen_new_tree_log(self, new_trees: pd.DataFrame) -> None:
        """
        Generate a log message for the new trees added to the stemmap.
        """
        if new_trees.shape[0] > 0:
            mean_diameter = new_trees["diameter"].mean()
            mean_x = new_trees["x"].mean()
            mean_y = new_trees["y"].mean()
            mean_z = new_trees["z"].mean()
            new_tree_log = f"Added {new_trees.shape[0]} new trees to the stemmap with mean diameter: {mean_diameter}, mean x: {mean_x}, mean y: {mean_y}, mean z: {mean_z}"
        else:
            new_tree_log = "No new trees added to the stemmap"

        logger.info(new_tree_log)

    def align_sub_stemmaps(
        self, old_stemmap: pd.DataFrame, new_stemmap: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Identify existing trees from the old stemmap that are within the new stemmap.
        Uses DBSCANN to cluster the trees and then aligns the clusters to the new stemmap.
        """
        # Assign 'old' and 'new' labels to the 'status' column
        old_stemmap["status"] = "old"
        new_stemmap["status"] = "new"
        current_timestamp = max(new_stemmap["timestamp"])

        # Convert x,y,z positions from both stem maps to ENU coordinates for DBSCANN
        old_stem_origin = old_stemmap.iloc[0][["lat","lon","elev"]].to_numpy()
        
        e_new,n_new,u_new = pm.geodetic2enu(
            lat=new_stemmap["lat"].to_numpy(),
            lon=new_stemmap["lon"].to_numpy(),
            h=new_stemmap["elev"].to_numpy(),
            lat0=old_stem_origin[0],
            lon0=old_stem_origin[1],
            h0=old_stem_origin[2]
            )
        new_stemmap["x"] = e_new
        new_stemmap["y"] = n_new
        new_stemmap["z"] = u_new
        # Concatenate the old and new stemmaps
        combined_stemmap = pd.concat([old_stemmap, new_stemmap], ignore_index=True).reset_index(drop=True)
        # Perform DBSCAN clustering
        dbscann = DBSCAN(
            eps=self.post_processing_dbscan_args["eps"],
            min_samples=self.post_processing_dbscan_args["min_samples"],
        )
        # TODO might need to scale 'diameter' to be within the same range as 'x' and 'y'
        labels = dbscann.fit_predict(combined_stemmap[["x", "y", "diameter"]])
        """
        For each label, perform the following:
        1. Identify the label as containing both 'old' and 'new' trees: 
            if the label contains both 'old' and 'new' trees, increase the 'old' tree's count by 1
            then aggregate the trees by 'median' and 'min' timestamp.
         
        2. Identify the label as containing only 'old' trees:
            if the label contains only 'old' trees, then mark the trees 'harvest_timestamp' as current_timestamp
        
        3. Identify the label as containing only 'new' trees:
            if the label contains only 'new' trees, then do nothing
        """
        update_count = 0
        harvest_count = 0
        new_count = 0
        combined_stemmap["harvest_timestamp"] = None
        to_merge = []

        for label, aligned_group in combined_stemmap.groupby(labels):
            status_list = aligned_group["status"].unique()

            # Process New and Seen Trees
            if "old" in status_list:

                # Update Matched Trees
                if "new" in status_list:
                    # get the original tree ids
                    id = aligned_group[aligned_group["status"]=="old"].iloc[0]["id"]

                    processed = aligned_group.agg(
                        {
                            "x":"median",
                            "y":"median",
                            "z":"min",
                            "lat":"median",
                            "lon": "median",
                            "elev":"min",
                            "diameter":"median",
                            "timestamp":"min",
                            "count":"sum"
                        }
                    ).to_frame().T.reset_index(drop=True)
                    processed["id"] = id
                    processed["status"] = "old"
                    processed["harvest_timestamp"] = None
                    update_count +=1
                
                # Process Harvested Trees
                else:
                    aligned_group["harvest_timestamp"] = current_timestamp
                    processed = aligned_group
                    harvest_count += processed.shape[0]


            
            # Process New and Unseen Trees
            else:
                processed = aligned_group
                processed["harvest_timestamp"] = None
                new_count += 1

            to_merge.append(processed)
            
        alignment_info = {
            "update_count": update_count,
            "harvest_count": harvest_count,
            "new_count": new_count,
        }

        tree_set_processed = pd.concat(to_merge).reset_index(drop=True)
        tree_set_processed["geometry"] = tree_set_processed.apply(lambda x: Point(x.lon,x.lat),axis=1)
        return tree_set_processed, alignment_info

    def generate_viewshed_map(self, positions: List[SMPosition]) -> Polygon:
        """
        Generate a viewshed map from a list of positions.
        """
        viewshed_list = []
        for position in positions:
            transformed_viewshed: Polygon = self.transform_viewshed(
                deepcopy(self.camera_viewshed), position
            )
            viewshed_list.append(transformed_viewshed)

        viewshed_map = unary_union(viewshed_list)
        return viewshed_map
    
    def rect_local(self,point:np.ndarray,origin:np.ndarray):

        lat,lon,alt = pm.enu2geodetic(point[:,0],point[:,1],np.zeros_like(point[:,1]),lat0=origin[0],lon0=origin[1],h0=origin[2])

        return np.round(np.vstack((lon,lat)).T,8)
    
    def transform_viewshed(self, viewshed: Polygon, position: SMPosition) -> Polygon:
        """
        Rotate the viewshed polygon to match the azimuth of the camera in compass coordinates.
        """

        polygon_coords = viewshed.exterior.coords
        
        # Step 1. Rotate the viewshed polygon to match the azimuth of the camera in compass coordinates.
        # Convert compass azimuth to degrees
        compass_azimuth = position.azimuth
        cartesian_angle = (90 - compass_azimuth) % 360
        cartesian_angle = (360 - compass_azimuth) % 360

        to_rotate = cartesian_angle + self.cam_info["fov_h"]/2
        # Rotate the viewshed polygon
        rotated_viewshed = rotate(
            geom=viewshed,
            angle=-cartesian_angle,
            origin=(0,0),
            use_radians=False,
        )
        _rect_local = partial(self.rect_local,origin=np.array([position.lat,position.lon,position.elevation]))
        # Step 2. Translate the viewshed polygon to match the position of the camera in cartesian coordinates.
        rectified_viewshed = transform(
            geometry=rotated_viewshed,
            transformation=_rect_local)
        
        return rectified_viewshed

    def generate_viewshed_polygon(self, distance, fov) -> Polygon:
        # Calculate half of the field of view angle
        half_fov = fov / 2

        thetas = np.linspace(-half_fov, half_fov, 20).tolist()

        # Generate the polygon
        vertices = [(0, 0)]
        for theta in thetas:
            x = distance * np.cos(np.deg2rad(theta))
            y = distance * np.sin(np.deg2rad(theta))
            vertices.append((x, y))

        # Create the polygon
        polygon = Polygon(vertices)

        return polygon

    def process_and_cluster_predictions(
        self,
        trees: Union[List[dict], pd.DataFrame],
        eps=0.5,
        min_samples=1,
        dbh_max=1.5,
        dbh_min=0.05,
    ) -> Union[pd.DataFrame, None]:
        """
        Process and cluster predictions based on spatial coordinates and diameter.

        See:
            - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

        Parameters:
        - predictions (list): List of prediction dictionaries.
        - eps (float): DBSCAN epsilon parameter.
        - min_samples (int): DBSCAN min_samples parameter.
        - dbh_max (float): Maximum diameter for filtering.
        - dbh_min (float): Minimum diameter for filtering.

        Returns:
        - pd.Dataframe : Processed and clustered data.
        """
        if isinstance(trees, list):
            data = pd.DataFrame(trees)

        # Filter out bad dbh
        data_filtered_dbh = data[
            (data["diameter"] < dbh_max) & (data["diameter"] > dbh_min)
        ]

        log = f"Filtered {data.shape[0] - data_filtered_dbh.shape[0]} trees based on diameter"
        logger.info(log)

        if data_filtered_dbh.shape[0] > 0:
            print("CLUSTERING")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            to_dbscan = data_filtered_dbh[["x", "y"]].to_numpy()
     
      
            tree_labels = dbscan.fit_predict(to_dbscan)

            # Use groupby and agg for a more concise and efficient way to get median and earliest_observation
            out = data_filtered_dbh.groupby(tree_labels).agg(
                {
                    "x": "median",
                    "y": "median",
                    "z": "median",
                    "lat":"median",
                    "lon":"median",
                    "elev":"median",
                    "diameter": "median",
                    "timestamp": "min",
                }
            )

            distill_log = f"Distilled {len(out)} trees from {len(data)} predictions with parameters: eps={eps} [m], min_samples={min_samples} [int]"
            logger.info(distill_log)

            return out

        return None

    def event_process_staged_trees(
        self, positions: List[SMPosition]
    ) -> Union[pd.DataFrame, None]:
        """
        Process staged trees and add them to the stemmap.
        """
        if self.staged_predictions:
            new_trees = self.add_trees(self.staged_predictions, positions)
            self.staged_predictions = []
            if new_trees is not None:
                return new_trees
            else:
                return None
        else:
            raise NoTreesFound
        
    def manual_process_staged_trees(self) -> pd.DataFrame:
     
        processed_trees: pd.DataFrame = self.process_and_cluster_predictions(
            self.staged_predictions, **self.pre_processing_dbscan_args
        )
        log = f"Distilled {len(processed_trees)} trees from {len(self.staged_predictions)} predictions"
        logger.info(log)

        processed_trees["count"] = 1
        processed_trees["id"] = [uuid.uuid4().hex for _ in range(len(processed_trees))]
        self.staged_predictions = []
        return processed_trees
    
    def stage_tree(self, prediction: dict) -> None:
        """
        Stage a tree for processing.
        """
        if isinstance(prediction, list):
            self.staged_predictions.extend(prediction)
        self.staged_predictions.append(prediction)

    def get_raw(self) -> pd.DataFrame:
        """
        Get the raw stemmap data.
        """
        df = pd.DataFrame(self.staged_predictions)
        return df


