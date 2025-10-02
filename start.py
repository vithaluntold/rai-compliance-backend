#!/usr/bin/env python3
"""
Startup script for Render deployment to prevent early exit
"""
import os
import signal
import sys
import uvicorn
from minimal import app

def signal_handler(sig, frame):
    print('Gracefully shutting down...')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    
    # Run with explicit configuration to prevent early exit
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=30,
        access_log=True,
        use_colors=True
    )