import logging
import os
from types import SimpleNamespace
from typing import Optional

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from supabase import Client, create_client

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    client = None
    db = None

    @classmethod
    def initialize(cls):
        if cls.client is None:
            uri = os.getenv("MONGO_URI")
            dbname = os.getenv("MONGO_DBNAME")
            cls.client = MongoClient(uri, server_api=ServerApi('1'))
            cls.db = cls.client[dbname]
            try:
                cls.client.admin.command('ping')
                logger.info("Connected to MongoDB")
            except Exception as e:
                logger.error(f"MongoDB connection failed: {e}")
                raise

    @classmethod
    def get_db(cls):
        if cls.client is None:
            cls.initialize()
        return cls.db

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("MongoDB connection closed")


class SupabaseDB:
    client: Optional[Client] = None
    table_name = "ctech_faculties"

    @classmethod
    def initialize(cls):
        if cls.client is None:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                raise RuntimeError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set")

            cls.client = create_client(supabase_url, supabase_key)

            try:
                cls.client.table(cls.table_name).select("id").limit(1).execute()
                logger.info("Connected to Supabase")
            except Exception as e:
                logger.error(f"Supabase connection failed: {e}")
                raise

    @classmethod
    def _ensure_client(cls):
        if cls.client is None:
            cls.initialize()

    @classmethod
    def _row_to_face(cls, row):
        if not row:
            return None
        return SimpleNamespace(**row)

    @classmethod
    def get_session(cls):
        raise NotImplementedError("SupabaseDB does not use SQLAlchemy sessions")

    @classmethod
    def get_face_by_reg_no(cls, reg_no: str):
        cls._ensure_client()
        try:
            response = (
                cls.client.table(cls.table_name)
                .select("id, reg_no, name, embedding")
                .eq("reg_no", reg_no)
                .limit(1)
                .execute()
            )
            rows = response.data or []
            return cls._row_to_face(rows[0] if rows else None)
        except Exception as e:
            logger.error(f"Error in get_face_by_reg_no: {e}")
            raise

    @classmethod
    def insert_or_update_face(cls, reg_no: str, name: str, embedding: list):
        cls._ensure_client()
        try:
            cls.client.table(cls.table_name).upsert(
                {
                    "reg_no": reg_no,
                    "name": name,
                    "embedding": embedding,
                },
                on_conflict="reg_no",
            ).execute()
            return cls.get_face_by_reg_no(reg_no)
        except Exception as e:
            logger.error(f"Error in insert_or_update_face: {e}")
            raise

    @classmethod
    def get_all_faces(cls):
        cls._ensure_client()
        try:
            response = (
                cls.client.table(cls.table_name)
                .select("reg_no, name")
                .order("name")
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Error in get_all_faces: {e}")
            raise

    @classmethod
    def delete_face(cls, reg_no):
        cls._ensure_client()
        try:
            response = (
                cls.client.table(cls.table_name)
                .delete()
                .eq("reg_no", reg_no)
                .execute()
            )
            return len(response.data or [])
        except Exception as e:
            logger.error(f"Error in delete_face: {e}")
            raise

    @classmethod
    def get_face_count(cls):
        cls._ensure_client()
        try:
            response = cls.client.table(cls.table_name).select("id", count="exact").execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"Error in get_face_count: {e}")
            raise

    @classmethod
    def close(cls):
        if cls.client:
            cls.client = None
            logger.info("Supabase connection closed")


PostgresDB = SupabaseDB
