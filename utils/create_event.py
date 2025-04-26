from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from .db import Database
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, time

class Event(BaseModel):
    event_name: str
    event_desc: str
    event_location: str
    event_sdate: date
    event_stime: time
    event_edate: date
    event_etime: time
    attendees: List[str] = Field(default_factory=list)



def create_event():
    pass