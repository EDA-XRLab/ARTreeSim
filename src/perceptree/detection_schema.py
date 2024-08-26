from pydantic import BaseModel,Field,model_validator,model_serializer,root_validator
from datetime import datetime
from typing import Optional,Union,Dict,Any
import numpy as np

import jsonpickle

class TreeProperties(BaseModel):
    x: float
    y: float
    z: float
    pixel_center: Optional[list]
    diameter: float
    time:datetime 


class CameraModel(BaseModel):
    c_col: float = Field(...,title="Principal point column")
    c_row: float = Field(...,title="Principal point row")
    f_col: float = Field(...,title="Focal length column [mm]")
    f_row: float = Field(...,title="Focal length row [mm]")    


class Image(BaseModel):
    rgb: Union[np.ndarray,str]
    depth: Union[np.ndarray,str]
    model: Union[CameraModel,Dict[str,Any]]
    timestamp: datetime

    @model_serializer(when_used='json')
    def serialize(self) -> Dict[str,Any]:
        return {k:jsonpickle.encode(v) for k,v in self.__dict__.items()}
        

    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def from_serial_json(cls, data:dict):
        data = {k:jsonpickle.decode(v, classes={'src.perceptree.detection_schema.CameraModel':CameraModel}) for k,v in data.items()}
        return cls(**data)
    