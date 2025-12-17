# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# --- Import your agent core ---
# Ensure agent_core.py is in the same folder as main.py
try:
    from agent_core import finance_agent
except ImportError:
    # Fallback for testing if agent_core isn't perfectly set up yet
    def finance_agent(query):
        return f"Mock response: You asked '{query}', but agent_core.py is missing or failed to import."

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Definition ---
app = FastAPI()

# --- CORS Configuration (Crucial for connecting Frontend to Backend) ---
origins = [
    "http://127.0.0.1:5500",  # Common Live Server port
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "null"                    # Allows opening index.html directly from file system
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all. Change this for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Model ---
class QueryRequest(BaseModel):
    query: str

# --- Routes ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Finance Agent API is running."}

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    """
    Endpoint called by the frontend.
    Receives JSON: {"query": "..."}
    Returns JSON: {"answer": "..."}
    """
    user_query = request.query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"Received query: {user_query}")

    try:
        # --- CALL YOUR AGENT HERE ---
        # This calls the main entry point function in your agent_core.py
        response_text = finance_agent(user_query)
        
        return {"answer": response_text}
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return a clean error to the frontend
        return {
            "answer": f"⚠️ An internal error occurred while processing your request: {str(e)}"
        }

# --- Entry Point (for running via 'python main.py') ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)