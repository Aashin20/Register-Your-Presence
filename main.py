from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from utils.db import Database, SupabaseDB
from pydantic import BaseModel
from typing import List, Optional
from datetime import date, time
import numpy as np
from haversine import haversine, Unit
from deepface import DeepFace
import tempfile
import shutil
from bson import ObjectId
import os
import datetime
import csv
import io
import re

@asynccontextmanager
async def lifespan(app: FastAPI):
    Database.initialize()
    SupabaseDB.initialize()
    yield
    Database.close()
    SupabaseDB.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://abcd.ngrok.io",
        "http://192.168.1.10:3000",
        "http://0.0.0.0",
        "https://ryp-frontend-nt04vd4m3-aashin20s-projects.vercel.app/",
        "localhost:3000",
        "abcd.ngrok.io",
        "192.168.1.10",
        "0.0.0.0",
        "ryp-frontend-nt04vd4m3-aashin20s-projects.vercel.app",
        "https://ryp-frontend-nu.vercel.app",  # Added your frontend domain
        "ryp-frontend-nu.vercel.app",
        "https://register-your-presence.onrender.com"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # WARNING: Don't use in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    reg_no: str = Form(...),
    user_lat: float = Form(...),
    user_lng: float = Form(...),
    selfie: UploadFile = File(...)
):
    temp_path = None
    try:
        if not selfie.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")

        mongo_db = Database.get_db()
        events_collection = mongo_db.EventDetails
        event = events_collection.find_one({
            '_id': ObjectId(event_id) if len(event_id) == 24 else event_id
        })
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
            
        attendees = event.get("attendees", [])
        for attendee in attendees:
            if attendee.get("reg_no") == reg_no:
                return {"status": "Already present", "message": "Attendance already marked for this event", "attendee_info": attendee}

        event_location = event.get("event_location", {})
        if (not event_location or event_location.get("type") != "Point" or not event_location.get("coordinates")):
            raise HTTPException(status_code=400, detail="Invalid event location format in database")
        
        event_coords = event_location["coordinates"]
        event_lng, event_lat = event_coords[0], event_coords[1]
        
        if not (-90 <= event_lat <= 90) or not (-180 <= event_lng <= 180):
            raise HTTPException(status_code=400, detail="Event coordinates out of valid range")
        if not (-90 <= user_lat <= 90) or not (-180 <= user_lng <= 180):
            raise HTTPException(status_code=400, detail="User coordinates out of valid range")
        
        user_loc = (user_lat, user_lng)
        event_loc = (event_lat, event_lng)
        distance_m = haversine(user_loc, event_loc, unit=Unit.METERS)
        ATTENDANCE_RADIUS_METERS = 100
        if distance_m > ATTENDANCE_RADIUS_METERS:
            raise HTTPException(status_code=400, detail=f"You are {distance_m:.0f}m away from the event location. Must be within {ATTENDANCE_RADIUS_METERS}m.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            await selfie.seek(0)
            content = await selfie.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        embed = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet",
            enforce_detection=True
        )
        
        if not embed or len(embed) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the selfie")
            
        result = embed[0]["embedding"]
        result = np.array(result)
        
        response = SupabaseDB.query("face_embeddings").match({"reg_no": reg_no}).execute()
        data = response.data

        if not data or len(data) == 0:
            raise HTTPException(status_code=404, detail="No matching registration found in database. Please contact admin.")

        stored_embedding_data = data[0]['embedding']
        
        if isinstance(stored_embedding_data, str):
            try:
                clean_str = stored_embedding_data.strip('[]').split(',')
                stored_embedding = np.array([float(val) for val in clean_str])
            except Exception as e:
                number_strings = re.findall(r'-?\d+\.?\d*', stored_embedding_data)
                stored_embedding = np.array([float(val) for val in number_strings])
        else:
            stored_embedding = np.array(stored_embedding_data)
        
        
        
        stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
        result_norm = result / np.linalg.norm(result)
        
        similarity = np.dot(stored_norm, result_norm)
        
        cosine_distance = 1 - similarity
        is_same_person = cosine_distance < 0.8

        if is_same_person:
            attendee_info = {
                "reg_no": reg_no,
                "name": data[0]['name'],
                "timestamp": datetime.datetime.utcnow()
            }
            
            events_collection.update_one(
                {'_id': ObjectId(event_id) if len(event_id) == 24 else event_id},
                {'$push': {'attendees': attendee_info}}
            )
            
            return {"status": "success", "message": "Attendance registered successfully"}
        else:
            raise HTTPException(status_code=403, detail="Face verification failed. Please try again.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
