#!/usr/bin/env python3
"""
Ultra-minimal test server - NO imports, NO dependencies
"""

import http.server
import socketserver
import os
import json

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "message": "Ultra-minimal server running"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"message": "Hello from ultra-minimal Python server!"}
            self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 8000))
    print(f"Starting ultra-minimal server on port {PORT}")
    
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Server running on http://0.0.0.0:{PORT}")
        httpd.serve_forever()