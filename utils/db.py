import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from supabase import create_client, Client

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

class SupabaseDB:
    client = None

    @classmethod
    def initialize(cls):
        if not cls.client:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
            
            try:
                cls.client = create_client(supabase_url, supabase_key)
                test_query = cls.client.table('face_embeddings').select("reg_no").limit(1).execute()
                print("Connected to Supabase successfully!")
            except Exception as e:
                print(f"Failed to connect to Supabase: {e}")
                raise e

    @classmethod
    def get_client(cls) -> Client:
        """Returns the Supabase client instance"""
        if not cls.client:
            cls.initialize()
        return cls.client
    
    @classmethod
    def query(cls, table_name):
        """Helper method to start a query on a table"""
        client = cls.get_client()
        return client.table(table_name).select("*")
    
    @classmethod
    def match_face(cls, embedding, threshold=0.6, match_count=1):
        """Helper method specifically for face matching using pgvector"""
        return cls.rpc(
            "match_face_vector", 
            {
                "query_embedding": embedding,
                "similarity_threshold": threshold,
                "match_count": match_count
            }
        ).execute()
    
    @classmethod
    def register_face(cls, reg_no, name, embedding):
        """Helper method to register a new face"""
        return cls.rpc(
            "register_face",
            {
                "p_reg_no": reg_no,
                "p_name": name,
                "p_embedding": embedding
            }
        ).execute()
    
    @classmethod
    def rpc(cls, function_name, params=None):
        """Helper method to call a stored procedure"""
        client = cls.get_client()
        return client.rpc(function_name, params or {})