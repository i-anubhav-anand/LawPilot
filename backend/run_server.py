#!/usr/bin/env python
import multiprocessing
import os
import uvicorn
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

def main():
    """Run the FastAPI server with multiple workers."""
    parser = argparse.ArgumentParser(description="Run the Legal AI Assistant server")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--host", type=str, help="Host to run on")
    parser.add_argument("--port", type=int, help="Port to run on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Calculate optimal number of workers if not specified
    # Using the common formula: (2 * num_cores) + 1
    # This allows full utilization of CPU cores while keeping one worker 
    # for handling I/O operations
    cores = multiprocessing.cpu_count()
    if args.workers:
        workers = args.workers
    else:
        workers = min((2 * cores) + 1, 8)  # Cap at 8 workers to avoid excessive memory usage
    
    # Get configuration from environment variables or args
    host = args.host or os.getenv("HOST", "0.0.0.0")
    port = args.port or int(os.getenv("PORT", "8000"))
    reload = args.reload
    
    # Set some environment variables for uvicorn
    os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure logs are output immediately
    
    print(f"✨ Starting Legal AI Assistant server")
    print(f"🖥️  Host: {host}, Port: {port}")
    print(f"👷 Workers: {workers} (of {cores} CPU cores)")
    print(f"🔄 Reload: {'Enabled' if reload else 'Disabled'}")
    print(f"📊 Available at: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🩺 Health Check: http://localhost:{port}/health")
    print(f"📈 Status: http://localhost:{port}/status")
    
    # Run with uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "log_level": "info",
        "timeout_keep_alive": 300,    # Increased from 120 to 300 seconds
        "limit_concurrency": 100,     # Limit concurrent connections to avoid overload
        "backlog": 2048,              # Allow more queued connections
        "h11_max_incomplete_event_size": 10 * 1024 * 1024,  # 10MB, up from 5MB
    }
    
    if reload:
        uvicorn_config["reload"] = True
        # When using reload, we can't use multiple workers
        print("⚠️  Reload mode enabled - using single worker")
        uvicorn.run(**uvicorn_config)
    else:
        uvicorn_config["workers"] = workers
        uvicorn.run(**uvicorn_config)

if __name__ == "__main__":
    main() 