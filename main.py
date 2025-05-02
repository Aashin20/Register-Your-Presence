from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from utils.db import Database, SupabaseDB
from pydantic import BaseModel
from typing import List, Optional
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
from pathlib import Path
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow to be less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelManager:
    _instance = None
    _model = None
    _initialized = False
    _model_weights = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def initialize(self):
        """Initialize the model during server startup"""
        if not self._initialized:
            logger.info("Initializing DeepFace model...")
            try:
                # Set DeepFace home directory
                os.environ["DEEPFACE_HOME"] = "/opt/render/.deepface"
                
                # Force model download and initialization during startup
                self._model = DeepFace
                
                # Create a dummy image for initialization
                import numpy as np
                from PIL import Image
                dummy_img = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
                dummy_path = "dummy.jpg"
                dummy_img.save(dummy_path)
                
                try:
                    # Force model loading with minimal computation
                    _ = self._model.represent(
                        img_path=dummy_path,
                        model_name="Facenet",
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    self._initialized = True
                    logger.info("DeepFace model initialized successfully")
                finally:
                    if os.path.exists(dummy_path):
                        os.remove(dummy_path)
                        
            except Exception as e:
                logger.error(f"Error initializing model: {str(e)}")
                raise
        return self._model

    def get_model(self):
        if not self._initialized:
            self.initialize()
        return self._model

    @staticmethod
    async def analyze_face(image_path):
        model_manager = ModelManager.get_instance()
        model = model_manager.get_model()
        
        backends = ['opencv', 'retinaface', 'mtcnn', 'ssd']
        last_error = None
        
        for backend in backends:
            try:
                result = model.represent(
                    img_path=image_path,
                    model_name="Facenet",
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True
                )
                if result and len(result) > 0:
                    return result[0]
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Backend {backend} failed: {str(e)}")
                continue
        
        raise ValueError(f"All face detection backends failed. Last error: {last_error}")

# Initialize ModelManager at the module level
model_manager = ModelManager.get_instance()

class GeoLocation(BaseModel):
    lat: float
    lng: float

class Event(BaseModel):
    event_name: str
    event_desc: str
    event_location: GeoLocation
    event_sdate: date
    event_stime: time
    event_edate: date
    event_etime: time
    attendees: Optional[List[str]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application"""
    logger.info("Starting up server...")
    try:
        # Initialize model during startup
        logger.info("Initializing model...")
        model_manager.initialize()
        logger.info("Model initialization complete")
        
        # Initialize databases
        Database.initialize()
        SupabaseDB.initialize()
        
        yield
        
    finally:
        logger.info("Shutting down server...")
        Database.close()
        SupabaseDB.close()

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
    event_dict = info.model_dump()
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

@app.post("/register_attendance")
async def register_attendance(
    event_id: str = Form(...),
    user_lat: float = Form(...),
    user_lng: float = Form(...),
    selfie: UploadFile = File(...)
):
    try:
        # Validate file type
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

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

        if not all(-90 <= x <= 90 for x in [event_lat, user_lat]) or \
           not all(-180 <= x <= 180 for x in [event_lng, user_lng]):
            raise HTTPException(status_code=400, detail="Coordinates out of valid range")

        # Distance calculation
        distance_m = haversine(
            (user_lat, user_lng),
            (event_lat, event_lng),
            unit=Unit.METERS
        )
        
        ATTENDANCE_RADIUS_METERS = 100
        if distance_m > ATTENDANCE_RADIUS_METERS:
            raise HTTPException(
                status_code=400,
                detail=f"You are {distance_m:.0f}m away from the event location. Must be within {ATTENDANCE_RADIUS_METERS}m."
            )

        # Face detection
        temp_file_path = None
        try:
            content = await selfie.read()
            fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
            with os.fdopen(fd, 'wb') as temp_file:
                temp_file.write(content)

            face_data = await ModelManager.analyze_face(temp_file_path)
            if not face_data:
                raise HTTPException(status_code=400, detail="No face detected in the image")

            embedding = face_data if isinstance(face_data, list) else face_data.tolist()

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")

        # Face matching
        try:
            SIMILARITY_THRESHOLD = 0.8
            response = SupabaseDB.get_client().rpc(
                'match_face_vector',
                {
                    'query_embedding': embedding,
                    'similarity_threshold': SIMILARITY_THRESHOLD,
                    'match_count': 1
                }
            ).execute()

            if not response.data or len(response.data) == 0:
                raise HTTPException(status_code=401, detail="Face not recognized as a registered user")

            match = response.data[0]
            best_match = match['reg_no']
            best_match_name = match['name']
            similarity_score = match['distance']

            if similarity_score > SIMILARITY_THRESHOLD:
                raise HTTPException(
                    status_code=401,
                    detail=f"Face not recognized with sufficient confidence (score: {similarity_score:.2f})"
                )

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
                }

            # Update attendance
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
                raise HTTPException(status_code=500, detail="Failed to update attendance")

            return {
                "status": "success",
                "message": f"Attendance registered successfully for {best_match_name}",
                "data": {
                    "reg_no": best_match,
                    "name": best_match_name,
                    "similarity_score": f"{similarity_score:.4f}"
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in face matching: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")