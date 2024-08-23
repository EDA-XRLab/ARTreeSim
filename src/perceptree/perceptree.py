import torch
import numpy as np
import os 
from os.path import dirname, abspath,join
os.environ["TORCHVISION_USE_FFMPEG"] = "0"
# import detectron2 utilities
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from typing import List,Dict,Union,Any
from pathlib import Path
import cv2
# local imports


class PredictionPixel:
    def __init__(self, model_output) -> None:
        keypoints = model_output.pred_keypoints.squeeze().numpy()
        self.mask = model_output.pred_masks.squeeze().numpy().flatten()
        self.stem_center = keypoints[0, :2][np.newaxis, :][:, ::-1]
        self.stem_bounds = np.vstack((keypoints[1, :2], keypoints[2, :2]))[:, ::-1]


class Perceptree:
    # copied from https://github.com/norlab-ulaval/PercepTreeV1/blob/main/demo_video.py
    # weights found at https://drive.google.com/u/0/uc?id=1Q5KV5beWVZXK_vlIED1jgpf4XJgN71ky&export=download
    def __init__(self) -> None:

        # torch.cuda.is_available()
        weights_dir = str(Path(__file__).parent/"weights")
        model_name = join(weights_dir,"ResNext-101_fold_01.pth")

        # All configurables are listed in /repos/detectron2/detectron2/config/defaults.py
        # https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references
        cfg = get_cfg()
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))

        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (tree)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1  
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        cfg.MODEL.MASK_ON = True

        cfg.OUTPUT_DIR = './output' 
        cfg.MODEL.WEIGHTS =  model_name
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.ROI_HEADS.NMS_THRES_TEST = .5

        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            self.device = torch.device("cuda:0")
        else:
            cfg.MODEL.DEVICE = "cpu"
            self.device = "cpu"

        # cfg.INPUT.MIN_SIZE_TEST = 0  # no resize at test time
        self.tree_metadata = MetadataCatalog.get("my_tree_dataset").set(thing_classes=["Tree"], keypoint_names=["kpCP", "kpL", "kpR", "AX1", "AX2"])
        # set detector

        self.model = DefaultPredictor(cfg)

        self.prediction_set = []   

    def enable_draw_visuals(self):
        self.vid_vis = VideoVisualizer(metadata=self.tree_metadata)

    def _process_depth(self,depth_image:np.ndarray) -> np.ndarray:
        depth_image = depth_image.flatten()
        depth_image[np.isinf(depth_image)] = np.nan
        depth_image[np.isnan(depth_image)] = 0
        depth_image[depth_image>30] = 0
        return depth_image

    def _estimate_max_frequency_depth(self, depth_image: np.ndarray, bin_size: float=.15) -> float:
        """
        Estimate the depth value with the maximum frequency from the depth image.

        Args:
            depth_image (np.ndarray): Depth measurement produced by the stereoview ZED camera.
            bin_size (float): Bin depth increments for the measurement algorithm.

        Returns:
            np.ndarray: Estimated depth value with the maximum frequency.
        """
        min_depth = np.floor(np.nanmin(depth_image))
        max_depth = np.ceil(np.nanmax(depth_image))
        bin_range = np.arange(min_depth, max_depth + bin_size, bin_size)
        valid_measurements = depth_image.flatten()
        valid_measurements = valid_measurements[~np.isnan(valid_measurements)]
        valid_measurements = valid_measurements[valid_measurements > 0]
        bin_indices = np.digitize(valid_measurements, bin_range) - 1
        bins = np.bincount(bin_indices, minlength=len(bin_range) - 1)

        # Find the bin with the maximum frequency
        max_frequency_bin = np.argmax(bins)
        max_frequency_depth = bin_range[max_frequency_bin]

        return max_frequency_depth

    def matrix2world(self,column:np.ndarray,row:np.ndarray,distance:np.ndarray,cam_model:dict) -> np.ndarray:
        c_col = cam_model["c_col"]
        c_row = cam_model["c_row"]
        f_col = cam_model["f_col"]
        f_row = cam_model["f_row"]
        if not isinstance(distance,np.ndarray):
            distance = np.full_like(column,fill_value=distance)

        camera_matrix = np.array([[f_col,0,c_col],[0,f_row,c_row],[0,0,1]])
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        data_matrix = np.hstack((column[:,np.newaxis],row[:,np.newaxis],np.ones_like(column)[:,np.newaxis])).T

        normalized_data = camera_matrix_inv @ data_matrix
        return normalized_data * distance


    def _process_prediction(self,
        prediction:PredictionPixel,
        depth:np.ndarray,
        camera_model:dict,
        timestamp: str) -> Union[dict,None]:

        depth_masked = depth[prediction.mask]
        try:
            distance = self._estimate_max_frequency_depth(depth_masked,bin_size=.15)

        except ValueError:
            return None
    
        if distance <.1:
            return None

        dbh_bounds_world = self.matrix2world(
            column = prediction.stem_bounds[:,1],
            row = prediction.stem_bounds[:,0],
            distance = distance,
            cam_model = camera_model
        ).T
        dbh = np.linalg.norm(dbh_bounds_world[0] - dbh_bounds_world[1])
        stem_center_world = self.matrix2world(
            column = prediction.stem_center[:,1],
            row = prediction.stem_center[:,0],
            distance = distance,
            cam_model = camera_model
        ).T
        # round stem_center_world to 3 decimal places
        stem_center_world = np.round(stem_center_world,3)
        dbh = np.round(dbh,3)
        x,y,z = stem_center_world.squeeze().tolist()

        processed_prediction = {"x":x,"y":y,"z":z,"diameter":dbh,"time":timestamp,"pixel_center":prediction.stem_center.squeeze().tolist()}
        
        return processed_prediction

    def predict(self,rgb,depth,model,timestamp,**kwargs) -> Union[List[dict],List[Any]]:
        out = []
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        predictions_raw = self.model(rgb)
        predictions_raw = predictions_raw["instances"].to("cpu")
        predictions_raw_pixel = [
            PredictionPixel(predictions_raw[idx]) for idx in range(len(predictions_raw)) ]
        depth_processed = self._process_depth(depth)
        for prediction_raw in predictions_raw_pixel:
            prediction = self._process_prediction(
                prediction=prediction_raw,
                depth=depth_processed,
                camera_model=model,
                timestamp=timestamp)
            if prediction is not None:
                out.append(prediction)
        return out
