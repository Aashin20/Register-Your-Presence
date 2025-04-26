from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from utils.db import init_postgres, init_mongo

from fastapi import FastAPI
from contextlib import asynccontextmanager
from utils.db import Database, PostgresDB

@asynccontextmanager
async def lifespan(app: FastAPI):
    Database.initialize()
    PostgresDB.initialize()
    yield
    Database.close()
    PostgresDB.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/login")
async def login(email: str, password: str, request: Request):
    if email == "admin@gmail.com" and password == "admin":
        return {"message": "Login successful"}
    else:
        return {"message": "Invalid credentials"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")