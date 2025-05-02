from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from utils.db import Database, SupabaseDB
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import date, time
import numpy as np
from haversine import haversine, Unit
from deepface import DeepFace
import tempfile
from bson import ObjectId
import os
import datetime
import csv
import io
import logging
import asyncio
from pathlib import Path
import tensorflow as tf
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to be less verbose and limit memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5  # Limit GPU memory usage
tf_session = tf.compat.v1.Session(config=tf_config)
tf.compat.v1.keras.backend.set_session(tf_session)

# Constants
MAX_IMAGE_SIZE = 1024 * 1024 * 5  # 5MB
FACE_DETECTION_TIMEOUT = 30  # 30 seconds
ATTENDANCE_RADIUS_METERS = 100
SIMILARITY_THRESHOLD = 0.8

class ModelManager:
    _instance = None
    _model = None
    _initialized = False
    _preferred_backend = 'retinaface'  # Set a single preferred backend

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def initialize(self):
        """Initialize the model during server startup, but with memory optimizations"""
        if not self._initialized:
            logger.info("Initializing DeepFace model...")
            try:
                # Make sure the directory exists
                os.makedirs(os.environ.get("DEEPFACE_HOME", "./deepface_models"), exist_ok=True)
                
                # Just store the reference, don't preload any models
                self._model = DeepFace
                self._initialized = True
                logger.info("DeepFace model initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing model: {str(e)}")
                raise
        return self._model

    def get_model(self):
        if not self._initialized:
            self.initialize()
        return self._model

    @staticmethod
    def convert_to_list(embedding):
        """Convert embedding to list format regardless of input type"""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        elif isinstance(embedding, (float, np.float32, np.float64)):
            # For single float values, create a list with that value
            return [float(embedding)]
        else:
            raise ValueError(f"Unexpected embedding type: {type(embedding)}")

    @staticmethod
    async def analyze_face(image_path):
        model_manager = ModelManager.get_instance()
        model = model_manager.get_model()
        
        # Use a single preferred backend with a timeout
        try:
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: model.represent(
                    img_path=image_path,
                    model_name="Facenet",
                    detector_backend=ModelManager._preferred_backend,
                    enforce_detection=True,
                    align=True
                )
            )
            
            if result and len(result) > 0:
                # Convert the embedding to the correct format
                embedding = result[0]
                return ModelManager.convert_to_list(embedding)
            
            raise ValueError("No face detected in the image")
        
        except Exception as e:
            # If the preferred backend fails, try a fallback
            try:
                logger.warning(f"Preferred backend failed: {str(e)}, trying fallback")
                result = await loop.run_in_executor(
                    None,
                    lambda: model.represent(
                        img_path=image_path,
                        model_name="Facenet",
                        detector_backend="opencv",  # Fallback to opencv
                        enforce_detection=True,
                        align=True
                    )
                )
                
                if result and len(result) > 0:
                    # Convert the embedding to the correct format
                    embedding = result[0]
                    return ModelManager.convert_to_list(embedding)
            except Exception as fallback_error:
                logger.error(f"Fallback detection failed: {str(fallback_error)}")
                
            raise ValueError(f"Face detection failed: {str(e)}")

# Initialize ModelManager at the module level
model_manager = ModelManager.get_instance()

class GeoLocation(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

class Event(BaseModel):
    event_name: str
    event_desc: str
    event_location: GeoLocation
    event_sdate: date
    event_stime: time
    event_edate: date
    event_etime: time
    attendees: Optional[List[Dict[str, str]]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    logger.info("Starting up server...")
    try:
        # Initialize model during startup - but don't preload everything
        logger.info("Initializing model...")
        model_manager.initialize()
        logger.info("Model initialization complete")
        
        # Initialize databases
        Database.initialize()
        SupabaseDB.initialize()
        
        yield
        
    finally:
        logger.info("Shutting down server...")
        # Clean up resources
        Database.close()
        SupabaseDB.close()
        # Clean up TensorFlow session
        tf.keras.backend.clear_session()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

@app.post("/login")
async def login(email: str, password: str, request: Request):
    if email == "admin@gmail.com" and password == "admin":
        return {"message": "Login successful"}
    else:
        return {"message": "Invalid credentials"}

@app.post("/create_event")
async def create_event(info: Event):
    db = Database.get_db().EventDetails
    exists = db.find_one({"event_name": info.event_name})
    if exists:
        return {"status": "Event already exists"}
    
    # Change from model_dump() to dict() for Pydantic v1 compatibility
    event_dict = info.dict()
    
    for field in ['event_sdate', 'event_edate']:
        event_dict[field] = event_dict[field].isoformat()
    for field in ['event_stime', 'event_etime']:
        event_dict[field] = event_dict[field].isoformat()
    event_dict['event_location'] = {
        "type": "Point",
        "coordinates": [event_dict['event_location']['lng'], event_dict['event_location']['lat']]
    }
    db.insert_one(event_dict)
    return {"status": "Event created successfully"}

@app.get("/active_events")
async def get_active_events():
    db = Database.get_db().EventDetails
    cursor = db.find(
        {"event_sdate": {"$gte": date.today().isoformat()}},
        {
            "event_name": 1,
            "event_sdate": 1,
            "event_stime": 1,
            "attendees": 1,
            "_id": 1
        }
    )
    event_list = []
    for event in cursor:
        event_summary = {
            "_id": str(event["_id"]),
            "event_name": event.get("event_name"),
            "event_sdate": event.get("event_sdate"),
            "event_stime": event.get("event_stime"),
            "attendees_count": len(event.get("attendees", []) or [])
        }
        event_list.append(event_summary)
    return {"events": event_list}

@app.get("/past_events")
async def get_past_events():
    db = Database.get_db().EventDetails
    cursor = db.find(
        {"event_sdate": {"$lt": date.today().isoformat()}},
        {
            "event_name": 1,
            "event_sdate": 1,
            "event_stime": 1,
            "attendees": 1,
            "_id": 1
        }
    )
    event_list = []
    for event in cursor:
        event_summary = {
            "_id": str(event["_id"]),
            "event_name": event.get("event_name"),
            "event_sdate": event.get("event_sdate"),
            "event_stime": event.get("event_stime"),
            "attendees_count": len(event.get("attendees", []) or [])
        }
        event_list.append(event_summary)
    return {"events": event_list}

@app.get("/available_events")
async def get_active_events_full():
    db = Database.get_db().EventDetails
    cursor = db.find(
        {"event_sdate": {"$gte": date.today().isoformat()}},
        {
            "event_name": 1,
            "event_desc": 1,
            "event_location": 1,
            "event_sdate": 1,
            "event_stime": 1,
            "event_edate": 1,
            "event_etime": 1,
            "attendees": 1,
        }
    )
    events = []
    for event in cursor:
        event["_id"] = str(event["_id"])
        event["attendees"] = event.get("attendees") or []
        events.append(event)
    return {"events": events}

@app.get("/event/{event_id}")
async def get_event_details(event_id: str):
    try:
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        event = events_collection.find_one({
            '_id': ObjectId(event_id) if len(event_id) == 24 else event_id
        })
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        event['_id'] = str(event['_id'])
        return event
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching event details: {str(e)}")

@app.get("/event/{event_id}/attendance/download")
async def download_attendance(event_id: str):
    try:
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        event = events_collection.find_one({
            '_id': ObjectId(event_id) if len(event_id) == 24 else event_id
        })
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Registration No', 'Name', 'Timestamp'])
        
        for attendee in event.get('attendees', []):
            writer.writerow([
                attendee.get('reg_no', ''),
                attendee.get('name', ''),
                attendee.get('timestamp', '')
            ])
        
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename=attendance-{event_id}.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading attendance: {str(e)}")

# Background task for face processing
async def process_face_async(
    event_id: str,
    user_lat: float,
    user_lng: float,
    temp_file_path: str,
    event: Dict[str, Any]
):
    try:
        # Face detection
        embedding = await ModelManager.analyze_face(temp_file_path)
        if not embedding:
            logger.warning("No face detected in the image")
            return None, "No face detected in the image"
        
        # Match face with database
        response = SupabaseDB.get_client().rpc(
            'match_face_vector',
            {
                'query_embedding': embedding,
                'similarity_threshold': SIMILARITY_THRESHOLD,
                'match_count': 1
            }
        ).execute()

        if not response.data or len(response.data) == 0:
            return None, "Face not recognized as a registered user"

        match = response.data[0]
        best_match = match['reg_no']
        best_match_name = match['name']
        similarity_score = match['distance']

        if similarity_score > SIMILARITY_THRESHOLD:
            return None, f"Face not recognized with sufficient confidence (score: {similarity_score:.2f})"

        # Check for duplicate attendance
        attendees = event.get('attendees', [])
        if attendees and any(
            isinstance(attendee, dict) and 
            attendee.get('reg_no') == best_match 
            for attendee in attendees
        ):
            return {
                "status": "already_registered",
                "message": f"Attendance already registered for {best_match_name}",
                "data": {
                    "reg_no": best_match,
                    "name": best_match_name,
                    "similarity_score": f"{similarity_score:.4f}"
                }
            }, None

        # Update attendance
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        update_result = events_collection.update_one(
            {'_id': event['_id']},
            {
                '$addToSet': {
                    'attendees': {
                        'reg_no': best_match,
                        'name': best_match_name,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                }
            }
        )

        if update_result.modified_count == 0:
            return None, "Failed to update attendance"

        return {
            "status": "success",
            "message": f"Attendance registered successfully for {best_match_name}",
            "data": {
                "reg_no": best_match,
                "name": best_match_name,
                "similarity_score": f"{similarity_score:.4f}"
            }
        }, None

    except Exception as e:
        logger.error(f"Error in face processing: {str(e)}")
        return None, f"Error in face processing: {str(e)}"
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

@app.post("/register_attendance")
async def register_attendance(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    user_lat: float = Form(...),
    user_lng: float = Form(...),
    selfie: UploadFile = File(...)
):
    try:
        # Validate file type
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # Validate file size (prevent large uploads)
        content = await selfie.read(MAX_IMAGE_SIZE + 1)
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail=f"Image too large. Maximum size is {MAX_IMAGE_SIZE/1024/1024}MB")

        # Get event details and validate
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        event = events_collection.find_one({
            '_id': ObjectId(event_id) if len(event_id) == 24 else event_id
        })
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Location validation
        event_location = event.get("event_location", {})
        if not event_location or event_location.get("type") != "Point" or not event_location.get("coordinates"):
            raise HTTPException(status_code=400, detail="Invalid event location format")

        event_coords = event_location["coordinates"]
        event_lng, event_lat = event_coords[0], event_coords[1]

        # Distance calculation
        distance_m = haversine(
            (user_lat, user_lng),
            (event_lat, event_lng),
            unit=Unit.METERS
        )
        
        if distance_m > ATTENDANCE_RADIUS_METERS:
            raise HTTPException(
                status_code=400,
                detail=f"You are {distance_m:.0f}m away from the event location. Must be within {ATTENDANCE_RADIUS_METERS}m."
            )

        # Create a temporary file for face analysis
        fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
        with os.fdopen(fd, 'wb') as temp_file:
            temp_file.write(content)
        
        # Process face asynchronously with timeout
        try:
            # Schedule immediate face processing
            result, error = await asyncio.wait_for(
                process_face_async(event_id, user_lat, user_lng, temp_file_path, event),
                timeout=FACE_DETECTION_TIMEOUT
            )
            
            if error:
                raise HTTPException(status_code=400, detail=error)
            
            return result
            
        except asyncio.TimeoutError:
            # If processing takes too long, run it in the background
            # and return a pending status
            background_tasks.add_task(
                process_face_async, event_id, user_lat, user_lng, temp_file_path, event
            )
            
            return {
                "status": "processing",
                "message": "Your attendance is being processed. Please check back in a few moments."
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Endpoint to check the status of a pending attendance
@app.get("/attendance_status/{event_id}/{reg_no}")
async def check_attendance_status(event_id: str, reg_no: str):
    try:
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        event = events_collection.find_one({
            '_id': ObjectId(event_id) if len(event_id) == 24 else event_id
        })
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        attendees = event.get('attendees', [])
        for attendee in attendees:
            if attendee.get('reg_no') == reg_no:
                return {
                    "status": "registered",
                    "message": f"Attendance confirmed for {attendee.get('name')}",
                    "data": {
                        "reg_no": reg_no,
                        "name": attendee.get('name'),
                        "timestamp": attendee.get('timestamp')
                    }
                }
        
        return {
            "status": "pending",
            "message": "Attendance not yet registered for this user."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking attendance status: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
