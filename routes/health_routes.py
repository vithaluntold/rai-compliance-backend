import logging
import socket
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException

from services.ai import get_ai_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    ai_status = "unavailable"
    try:
        get_ai_service()
        ai_status = "available"
    except RuntimeError:
        pass  # Keep status unavailable

    return {
        "status": "healthy",
        "services": {
            "api": "up",
            "ai_service": "up" if ai_status == "available" else "down",
        },
    }


@router.get("/health/openai")
async def check_openai_health():
    """Check OpenAI API connectivity"""
    ai_svc = None
    ai_key = None
    ai_status = "unavailable"
    try:
        ai_svc = get_ai_service()
        ai_key = ai_svc.api_key  # Get key from the service instance
        ai_status = "available"
    except RuntimeError:
        pass  # Keep status unavailable and key None

    try:
        # Check DNS resolution
        api_host = urlparse("https://api.openai.com").netloc
        try:
            ip_address = socket.gethostbyname(api_host)
            dns_status = "healthy"
        except socket.gaierror as e:
            dns_status = f"error: {str(e)}"
            ip_address = None

        # Check HTTP connectivity
        http_status = "unknown"
        auth_header = {"Authorization": f"Bearer {ai_key}"} if ai_key else {}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models", headers=auth_header
                )
                http_status = response.status_code
        except Exception as e:
            http_status = f"error: {str(e)}"

        return {
            "status": (
                "healthy"
                if all(
                    [
                        dns_status == "healthy",
                        isinstance(http_status, int) and http_status == 200,
                        ai_status == "available",
                    ]
                )
                else "unhealthy"
            ),
            "details": {
                "dns_resolution": {"status": dns_status, "ip_address": ip_address},
                "http_connectivity": {"status": http_status},
                "ai_service": {"status": ai_status},
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
