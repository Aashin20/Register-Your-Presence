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
import face_recognition
import cv2
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

# Constants
# Constants
MAX_IMAGE_SIZE = 1024 * 1024 * 5  # 5MB
FACE_DETECTION_TIMEOUT = 30  # 30 seconds
ATTENDANCE_RADIUS_METERS = 100
SIMILARITY_THRESHOLD = 0.8  # Adjusted for face_recognition library

class ModelManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    @staticmethod
    async def analyze_face(image_path: str) -> List[float]:
        """Get face encoding from image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")  # Use 'cnn' for GPU
            
            if not face_locations:
                raise ValueError("No face detected in the image")

            if len(face_locations) > 1:
                logger.warning("Multiple faces detected in image, using the first one")

            # Get face encoding
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                raise ValueError("Failed to encode face")

            # Convert numpy array to list
            encoding = face_encodings[0].tolist()
            return encoding

        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            raise ValueError(f"Face analysis failed: {str(e)}")

    @staticmethod
    def compare_faces(encoding1: List[float], encoding2: List[float], tolerance: float = 0.6) -> float:
        """Compare two face encodings"""
        try:
            # Convert lists to numpy arrays
            np_encoding1 = np.array(encoding1)
            np_encoding2 = np.array(encoding2)
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(np_encoding1 - np_encoding2)
            
            # Convert distance to similarity score (inverse relationship)
            similarity = 1 - distance
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing faces: {str(e)}")
            raise
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
        model_manager = ModelManager.get_instance()
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
        
        # More robust ObjectId handling
        try:
            if len(event_id) == 24:
                # Try to convert to ObjectId
                id_query = ObjectId(event_id)
            else:
                # Use as string
                id_query = event_id
                
            # Log the query being attempted
            logger.info(f"Querying for event with ID: {id_query}")
            
        except Exception as id_error:
            # If ObjectId conversion fails, try as string
            logger.warning(f"Failed to convert to ObjectId: {str(id_error)}, using as string")
            id_query = event_id
        
        # Try multiple query approaches - first with ObjectId if possible
        event = None
        if len(event_id) == 24:
            try:
                event = events_collection.find_one({'_id': ObjectId(event_id)})
            except Exception:
                pass
                
        # If not found, try with string ID
        if not event:
            event = events_collection.find_one({'_id': event_id})
            
        # If still not found, try case-insensitive search
        if not event:
            event = events_collection.find_one({'_id': {'$regex': f'^{event_id}$', '$options': 'i'}})
        
        # Final check
        if not event:
            logger.error(f"Event not found with ID: {event_id}")
            raise HTTPException(status_code=404, detail=f"Event not found with ID: {event_id}")
        
        # Convert ObjectId to string for JSON serialization
        event['_id'] = str(event['_id'])
        return event
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error with traceback
        import traceback
        logger.error(f"Error fetching event: {str(e)}\n{traceback.format_exc()}")
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
    reg_no: str,
    temp_file_path: str,
    event: Dict[str, Any]
):
    try:
        logger.info(f"Processing attendance for reg_no: {reg_no}")
        
        # Get user data from Supabase
        user_data = SupabaseDB.get_client().table('face_embeddings') \
            .select('*') \
            .eq('reg_no', reg_no) \
            .execute()
        
        if not user_data.data or len(user_data.data) == 0:
            return None, f"No user found with registration number {reg_no}"

        user = user_data.data[0]
        stored_encoding = user.get('embedding')

        if not stored_encoding:
            return None, f"No face data found for registration number {reg_no}"

        # Check for duplicate attendance
        if event.get('attendees') and any(
            isinstance(attendee, dict) and 
            attendee.get('reg_no') == reg_no 
            for attendee in event['attendees']
        ):
            return {
                "status": "already_registered",
                "message": f"Attendance already registered for {user['name']}",
                "data": {
                    "reg_no": reg_no,
                    "name": user['name']
                }
            }, None

        try:
            # Get face encoding from selfie
            new_encoding = await ModelManager.analyze_face(temp_file_path)
            
            if not new_encoding:
                return None, "No face detected in the selfie"

            # Compare faces
            similarity_score = ModelManager.compare_faces(stored_encoding, new_encoding)
            logger.info(f"Face similarity score: {similarity_score:.4f}")

            if similarity_score < SIMILARITY_THRESHOLD:
                return None, f"Face verification failed. Score: {similarity_score:.2f}"

            # Update attendance in database
            attendance_record = {
                'reg_no': reg_no,
                'name': user['name'],
                'timestamp': datetime.datetime.now().isoformat(),
                'verification_score': float(similarity_score)
            }

            mongo_db = Database.get_db()
            events_collection = mongo_db.EventDetails
            
            update_result = events_collection.update_one(
                {'_id': event['_id']},
                {'$addToSet': {'attendees': attendance_record}}
            )

            if update_result.modified_count == 0:
                return None, "Failed to update attendance"

            return {
                "status": "success",
                "message": f"Attendance registered successfully for {user['name']}",
                "data": {
                    "reg_no": reg_no,
                    "name": user['name'],
                    "similarity_score": f"{similarity_score:.4f}",
                    "timestamp": attendance_record['timestamp']
                }
            }, None

        except Exception as face_error:
            logger.error(f"Error processing face: {str(face_error)}")
            return None, f"Error processing face: {str(face_error)}"

    except Exception as e:
        logger.error(f"Error in face processing: {str(e)}")
        return None, f"Error in face processing: {str(e)}"
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

@app.get("/verify_registration/{reg_no}")
async def verify_registration(reg_no: str):
    try:
        user_data = SupabaseDB.get_client().table('face_embeddings') \
            .select('reg_no,name') \
            .eq('reg_no', reg_no) \
            .execute()
        
        is_valid = bool(user_data.data and len(user_data.data) > 0)
        
        if is_valid:
            return {
                "valid": True,
                "name": user_data.data[0].get('name')
            }
        return {"valid": False}
        
    except Exception as e:
        logger.error(f"Error verifying registration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error verifying registration number"
        )
        
@app.post("/register_attendance")
async def register_attendance(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    reg_no: str = Form(...),
    user_lat: float = Form(...),
    user_lng: float = Form(...),
    selfie: UploadFile = File(...)
):
    try:
        logger.info(f"Received attendance registration request for event: {event_id}, reg_no: {reg_no}")
        
        # Input validation
        if not reg_no or not reg_no.strip():
            raise HTTPException(status_code=400, detail="Registration number is required")
        
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # Read and validate file size
        content = await selfie.read(MAX_IMAGE_SIZE + 1)
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Image too large. Maximum size is {MAX_IMAGE_SIZE/1024/1024}MB"
            )

        # Get event details
        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        
        try:
            event_query_id = ObjectId(event_id) if len(event_id) == 24 else event_id
            event = events_collection.find_one({'_id': event_query_id})
        except Exception as e:
            logger.error(f"Error querying event: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid event ID format")

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Validate event location
        event_location = event.get("event_location", {})
        if not event_location or \
           event_location.get("type") != "Point" or \
           not event_location.get("coordinates"):
            raise HTTPException(status_code=400, detail="Invalid event location format")

        # Extract coordinates and calculate distance
        event_coords = event_location["coordinates"]
        event_lng, event_lat = event_coords[0], event_coords[1]

        distance_m = haversine(
            (user_lat, user_lng),
            (event_lat, event_lng),
            unit=Unit.METERS
        )
        
        logger.info(f"User distance from event: {distance_m:.2f}m")
        
        if distance_m > ATTENDANCE_RADIUS_METERS:
            raise HTTPException(
                status_code=400,
                detail=f"You are {distance_m:.0f}m away from the event location. Must be within {ATTENDANCE_RADIUS_METERS}m."
            )

        # Create temporary file for face processing
        fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
        try:
            with os.fdopen(fd, 'wb') as temp_file:
                temp_file.write(content)
            
            # Process face with timeout
            result, error = await asyncio.wait_for(
                process_face_async(
                    event_id,
                    user_lat,
                    user_lng,
                    reg_no,
                    temp_file_path,
                    event
                ),
                timeout=FACE_DETECTION_TIMEOUT
            )
            
            if error:
                raise HTTPException(status_code=400, detail=error)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning("Face processing timeout, switching to background processing")
            # If processing takes too long, switch to background processing
            background_tasks.add_task(
                process_face_async,
                event_id,
                user_lat,
                user_lng,
                reg_no,
                temp_file_path,
                event
            )
            
            return {
                "status": "processing",
                "message": "Your attendance is being processed. Please check back in a few moments.",
                "data": {
                    "reg_no": reg_no,
                    "check_url": f"/attendance_status/{event_id}/{reg_no}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error during face processing: {str(e)}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=str(e))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
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
