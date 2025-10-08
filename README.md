# FastAPI Backend README

## 🚀 BlueCart ERP FastAPI Backend

A high-performance Python backend built with FastAPI for the BlueCart ERP system.

### 📋 Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **PostgreSQL Integration**: Full database support with SQLAlchemy ORM
- **JWT Authentication**: Secure user authentication and authorization
- **Automatic API Documentation**: Interactive docs at `/docs` and `/redoc`
- **Docker Support**: Complete containerization with Docker Compose
- **Comprehensive Testing**: Full test suite with pytest
- **Input Validation**: Pydantic schemas for data validation
- **CORS Support**: Ready for frontend integration

### 🛠️ Tech Stack

- **Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with SQLAlchemy 2.0
- **Authentication**: JWT with python-jose
- **Validation**: Pydantic v2
- **Testing**: pytest with httpx
- **Deployment**: Docker & Docker Compose

### 📁 Project Structure

```
backend/
├── main.py              # FastAPI application entry point
├── models.py            # SQLAlchemy database models
├── schemas.py           # Pydantic schemas for validation
├── crud.py              # Database operations
├── database.py          # Database connection and setup
├── auth.py              # Authentication and authorization
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
├── .env                 # Environment variables
├── test_api.py          # API tests
└── setup.py             # Setup and testing script
```

### 🚀 Quick Start

#### Option 1: Python Virtual Environment

1. **Create and activate virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

4. **Run setup script**:
   ```bash
   python setup.py
   ```

#### Option 2: Docker (Recommended)

1. **Start all services**:
   ```bash
   cd backend
   docker-compose up -d
   ```

2. **Access services**:
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - pgAdmin: http://localhost:5050 (admin@bluecart.com / admin123)

### 📊 API Endpoints

#### Health & Status
- `GET /` - Basic health check
- `GET /health` - Detailed health information

#### Shipments
- `POST /api/shipments` - Create new shipment
- `GET /api/shipments` - List all shipments (with pagination)
- `GET /api/shipments/{id}` - Get shipment by ID/tracking number
- `PUT /api/shipments/{id}` - Update shipment
- `DELETE /api/shipments/{id}` - Delete shipment
- `POST /api/shipments/{id}/events` - Add event to shipment

#### Analytics
- `GET /api/analytics/dashboard` - Get dashboard statistics

### 💡 API Usage Examples

#### Create a Shipment
```bash
curl -X POST "http://localhost:8000/api/shipments" \
  -H "Content-Type: application/json" \
  -d '{
    "sender_name": "John Doe",
    "sender_address": "123 Main St, City, State 12345",
    "receiver_name": "Jane Smith",
    "receiver_address": "456 Oak Ave, City, State 67890",
    "package_details": "Electronics - Laptop",
    "weight": 2.5,
    "dimensions": {
      "length": 40.0,
      "width": 30.0,
      "height": 5.0
    },
    "service_type": "express",
    "cost": 25.99
  }'
```

#### Get All Shipments
```bash
curl "http://localhost:8000/api/shipments?limit=10&skip=0"
```

#### Get Shipment by ID
```bash
curl "http://localhost:8000/api/shipments/SH12345678"
```

### 🧪 Testing

Run the test suite:
```bash
python -m pytest test_api.py -v
```

Run specific test:
```bash
python -m pytest test_api.py::test_create_shipment -v
```

### 🔧 Development

#### Start development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Database migrations (if using Alembic):
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### 🐳 Docker Commands

```bash
# Build and start services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild specific service
docker-compose build backend
```

### 🔒 Environment Variables

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=root
POSTGRES_DB=shipment_erp

# Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

### 🔍 Monitoring & Logging

The application includes:
- Health check endpoints for monitoring
- Structured logging
- Error handling and reporting
- Request/response logging in development

### 🚀 Deployment

#### Production Deployment with Docker:
```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

#### Environment Setup:
- Set `DEBUG=False`
- Use strong `SECRET_KEY`
- Configure proper database credentials
- Set up SSL certificates
- Configure reverse proxy (nginx)

### 🤝 Integration with Frontend

The FastAPI backend is designed to work with the Next.js frontend:

1. **CORS**: Configured to allow requests from `http://localhost:3000`
2. **API Routes**: RESTful endpoints that match frontend expectations
3. **Data Format**: JSON responses compatible with frontend models
4. **Authentication**: JWT tokens for secure API access

### 📈 Performance

- **Async Support**: FastAPI's async capabilities for high performance
- **Database Connection Pooling**: Efficient database connections
- **Caching**: Redis integration for caching (in docker-compose)
- **Pagination**: Built-in pagination for large datasets

### 🐛 Troubleshooting

#### Common Issues:

1. **Database Connection Error**:
   - Check PostgreSQL is running
   - Verify credentials in `.env`
   - Ensure database exists

2. **Import Errors**:
   - Activate virtual environment
   - Install requirements: `pip install -r requirements.txt`

3. **Port Already in Use**:
   - Change port in `.env` or docker-compose.yml
   - Kill existing processes: `lsof -ti:8000 | xargs kill`

### 📞 Support

For issues and questions:
1. Check the logs: `docker-compose logs backend`
2. Review API documentation: http://localhost:8000/docs
3. Run tests to verify setup: `python -m pytest test_api.py -v`

---

**🎉 Your FastAPI backend is ready for production!**