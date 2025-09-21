import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = 'ctech_faculties'
    
    id = Column(Integer, primary_key=True)
    reg_no = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)

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

class PostgresDB:
    engine = None
    SessionLocal = None

    @classmethod
    def initialize(cls):
        if cls.engine is None:
            postgres_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
            
            cls.engine = create_engine(
                postgres_url,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            cls.SessionLocal = scoped_session(sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=cls.engine
            ))
            
            try:
                with cls.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to PostgreSQL")
            except Exception as e:
                logger.error(f"PostgreSQL connection failed: {e}")
                raise

    @classmethod
    def get_session(cls):
        if cls.SessionLocal is None:
            cls.initialize()
        return cls.SessionLocal()

    @classmethod
    def get_face_by_reg_no(cls, reg_no: str):
        session = cls.get_session()
        try:
            return session.query(FaceEmbedding).filter(FaceEmbedding.reg_no == reg_no).first()
        finally:
            session.close()

    @classmethod
    def insert_or_update_face(cls, reg_no: str, name: str, embedding: list):
        session = cls.get_session()
        try:
            face = session.query(FaceEmbedding).filter(FaceEmbedding.reg_no == reg_no).first()
            if face:
                face.name = name
                face.embedding = embedding
                face.updated_at = func.now()
            else:
                face = FaceEmbedding(reg_no=reg_no, name=name, embedding=embedding)
                session.add(face)
            
            session.commit()
            return {'reg_no': face.reg_no, 'name': face.name, 'id': face.id}
        except Exception as e:
            session.rollback()
            logger.error(f"Error in insert_or_update_face: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def get_all_faces(cls):
        session = cls.get_session()
        try:
            faces = session.query(FaceEmbedding.reg_no, FaceEmbedding.name).order_by(FaceEmbedding.name).all()
            return [{'reg_no': face.reg_no, 'name': face.name} for face in faces]
        finally:
            session.close()

    @classmethod
    def delete_face(cls, reg_no):
        session = cls.get_session()
        try:
            deleted_count = session.query(FaceEmbedding).filter(FaceEmbedding.reg_no == reg_no).delete()
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error in delete_face: {e}")
            raise
        finally:
            session.close()

    @classmethod
    def get_face_count(cls):
        session = cls.get_session()
        try:
            return session.query(FaceEmbedding).count()
        finally:
            session.close()

    @classmethod
    def close(cls):
        if cls.SessionLocal:
            cls.SessionLocal.remove()
        if cls.engine:
            cls.engine.dispose()
            cls.engine = None
            cls.SessionLocal = None
            logger.info("PostgreSQL connection closed")