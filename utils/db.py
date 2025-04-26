import os
from dotenv import load_dotenv
import psycopg2
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

class Database:
    client = None
    db = None

    @classmethod
    def initialize(cls):
        if not cls.client:
            uri = os.getenv("MONGO_URI")
            dbname = os.getenv("MONGO_DBNAME")
            cls.client = MongoClient(uri, server_api=ServerApi('1'))
            cls.db = cls.client[dbname]
            try:
                cls.client.admin.command('ping')
                print("Connected to MongoDB!")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                raise e

    @classmethod
    def get_db(cls):
        if not cls.client:
            cls.initialize()
        return cls.db

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            print("MongoDB connection closed")
            cls.client = None
            cls.db = None

class PostgresDB:
    connection = None

    @classmethod
    def initialize(cls):
        if not cls.connection:
            user = os.getenv("PG_USER")
            password = os.getenv("PG_PASSWORD")
            host = os.getenv("PG_HOST")
            port = os.getenv("PG_PORT")
            dbname = os.getenv("PG_DBNAME")
            try:
                cls.connection = psycopg2.connect(
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                    dbname=dbname
                )
                print("Connected to Postgres!")
            except Exception as e:
                print(f"Failed to connect to Postgres: {e}")
                raise e

    @classmethod
    def get_connection(cls):
        if not cls.connection:
            cls.initialize()
        return cls.connection

    @classmethod
    def close(cls):
        if cls.connection:
            cls.connection.close()
            print("Postgres connection closed")
            cls.connection = None