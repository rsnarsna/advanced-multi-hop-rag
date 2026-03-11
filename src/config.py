import os
import asyncpg
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

# Neon PGVector Settings
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")

# Neo4j Settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", NEO4J_USERNAME)

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Mistral Settings
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class Config:
    _neo4j_graph = None
    _pg_pool = None

    @classmethod
    def get_neo4j_graph(cls):
        """Returns the LangChain Neo4jGraph integration instance."""
        if cls._neo4j_graph is None and NEO4J_URI:
            cls._neo4j_graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_DATABASE
            )
        return cls._neo4j_graph

    @classmethod
    async def get_pg_pool(cls):
        """Returns an asyncpg connection pool for Neon PGVector."""
        if cls._pg_pool is None and NEON_DATABASE_URL:
            if not NEON_DATABASE_URL:
                return None
            cls._pg_pool = await asyncpg.create_pool(
                dsn=NEON_DATABASE_URL,
                min_size=1,
                max_size=20 # PgBouncer on Neon can handle this
            )
        return cls._pg_pool

    @classmethod
    async def close_all(cls):
        """Closes all active database connection pools."""
        # Neo4jGraph handles its own connection lifecycle, so we only close PG
        if cls._pg_pool:
            await cls._pg_pool.close()

    @classmethod
    def get_llm(cls, temperature: float = 0.0):
        """Returns the configured LLM instance for extraction and reasoning."""
        from langchain_mistralai import ChatMistralAI
        # Using Mistral's flagship model
        return ChatMistralAI(
            model="mistral-large-latest", 
            temperature=temperature, 
            api_key=MISTRAL_API_KEY
        )

    @classmethod
    def get_embeddings(cls):
        """Returns the configured Embeddings instance."""
        from langchain_mistralai import MistralAIEmbeddings
        return MistralAIEmbeddings(
            model="mistral-embed", 
            api_key=MISTRAL_API_KEY
        )
