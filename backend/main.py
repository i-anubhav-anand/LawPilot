import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import psutil
import time
import asyncio
import threading

# Import routers
from app.api.router import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("app.log")  # Save to file
    ]
)
logger = logging.getLogger("main")

# Make sure all loggers use this configuration
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.INFO)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

app = FastAPI(title="Legal AI Assistant", description="AI Legal Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# Add middleware to track request processing times
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests (more than 1 second)
    if process_time > 1.0:
        logger.warning(f"SLOW REQUEST: {request.method} {request.url.path} took {process_time:.2f} seconds")
    
    return response

# Include routers
app.include_router(api_router, prefix="/api")
logger.info("API routers included")

# Track active document processing
active_processing = False

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to Legal AI Assistant"}

@app.get("/health")
async def health_check():
    """
    Special lightweight health check that always responds immediately.
    This endpoint is designed to be extremely fast and not rely on any database or processing.
    """
    # Don't even bother with logging to ensure this remains super lightweight
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def server_status():
    """
    Get the server status - lightweight endpoint that will always respond quickly.
    """
    # Get basic system information
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    # Get number of running workers
    process_count = len([p for p in psutil.process_iter() if "uvicorn" in p.name().lower()])
    
    # Get thread count for this process
    thread_count = threading.active_count()
    
    # Get event loop info
    loop = asyncio.get_event_loop()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "available_memory_mb": memory.available / (1024 * 1024),
        },
        "server": {
            "workers": process_count,
            "threads": thread_count,
            "uptime_seconds": time.time() - psutil.boot_time(),
        }
    }

@app.get("/logs", response_model=List[str])
async def get_logs(lines: Optional[int] = 100, component: Optional[str] = None):
    """
    Get the most recent log entries.
    
    Args:
        lines: Number of most recent log lines to return
        component: Filter by component name (e.g., "document_processor", "vector_store", "text_chunker")
    
    Returns:
        List of log lines
    """
    logger.info(f"Logs endpoint accessed: lines={lines}, component={component}")
    
    log_file = Path("app.log")
    if not log_file.exists():
        return ["No logs found"]
    
    try:
        # Read the log file
        with open(log_file, "r") as f:
            all_logs = f.readlines()
        
        # Filter by component if specified
        if component:
            filtered_logs = [line for line in all_logs if f" - {component} - " in line]
        else:
            filtered_logs = all_logs
        
        # Return the most recent logs, limited by the lines parameter
        return filtered_logs[-lines:]
        
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return [f"Error reading logs: {str(e)}"]

if __name__ == "__main__":
    logger.info("Starting server")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 