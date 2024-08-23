import redis
from typing import Optional,List
from datetime import datetime
import numpy as np
from .perceptree import Image,CameraModel,TreeProperties


class RedisHandler:
    """
    This class allows for interface with the perceptree microservice outside of the FastAPI application.

    """
    def __init__(self):
        """
        Initializes the IO class.
        """
        # Connect to the redis database hosted in the running container
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.camera_model: CameraModel = None

    def flush(self):
        """
        Flushes all data from the Redis database.
        """
        self.redis.flushall()

    def set_camera_model(self, f_col: float, f_row: float, c_col: float, c_row: float):
        """
        Set the camera model with the given parameters.

        Args:
            f_col (float): The focal length in the column direction.
            f_row (float): The focal length in the row direction.
            c_col (float): The principal point in the column direction.
            c_row (float): The principal point in the row direction.
        """
        self.camera_model = CameraModel(f_col=f_col, f_row=f_row, c_col=c_col, c_row=c_row)

    def add_image(self, rgb: np.ndarray, depth: np.ndarray, timestamp: Optional[datetime] = None):
        """
        Add an image to the system for prediction.

        Args:
            rgb (np.ndarray): The RGB image as a NumPy array.
            depth (np.ndarray): The depth image as a NumPy array. Units in meters.
            timestamp (Optional[datetime]): The timestamp of the image. If not provided, the current timestamp will be used.

        Raises:
            ValueError: If the camera model is not set or if there is an error adding the image to Redis.

        Example:
            >>> camera = get_camera_source()
            >>> redis_handler = RedisHandler()
            >>> cam_data: dict = camera.get_camera_data()
            >>> cam_data
            >>> {'f_col': 500, 'f_row': 500, 'c_col': 320, 'c_row': 240}
            >>> redis_handler.set_camera_model(**cam_data)
            >>> for rgb,depth,timestamp in camera.get_images():
                redis_handler.add_image(rgb,depth,timestamp)
        """
        if self.camera_model is None:
            raise ValueError("Camera model not set")
        if timestamp is None:
            timestamp = datetime.now()

        try:
            image = Image(rgb=rgb, depth=depth, model=self.camera_model, timestamp=timestamp)
            self.redis.lpush("images", image.model_dump_json())
        except Exception as e:
            raise ValueError(f"Error adding image to Redis: {str(e)}")
        
    def get_predictions(self, clear: bool = False) -> List[TreeProperties]:
            """
            Retrieves the predictions from the Redis database.

            Args:
                clear (bool, optional): Whether to clear the detections from the Redis database. Defaults to False.

            Returns:
                List[TreeProperties]: A list of TreeProperties objects representing the predictions.

            """
            
            detections = [TreeProperties.model_validate_strings(x) for x in self.redis.lrange("detections",0,-1)]
            if clear:
                self.redis.delete("detections")
            if not detections:
                return []
            return detections
