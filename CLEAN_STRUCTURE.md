# BlueCart ERP - Clean Backend Structure

## 📁 **Consolidated Backend Directory**

The backend has been cleaned up and consolidated into a single directory:

```
f:\ecom\bluecart-backend\
├── .env                    # Environment variables
├── main_simple.py          # Simple FastAPI server (currently running)
├── main_fastapi.py         # Advanced FastAPI server with database
├── main_postgres.py        # PostgreSQL-focused server
├── models.py               # Database models
├── schemas.py              # Pydantic schemas
├── auth.py                 # Authentication
├── crud.py                 # Database operations
├── database.py             # Database connection
├── requirements.txt        # Python dependencies
├── quick_test.py           # API testing script
├── test_*.py              # Various test files
├── database/              # Database setup files
│   ├── schema.sql
│   ├── setup-database.sql
│   └── connection.py
└── docker-compose.yml     # Docker configuration
```

## 🚀 **Current Setup**

- **Backend Server**: `http://localhost:8000` (running from `main_simple.py`)
- **Frontend Server**: `http://localhost:3000`
- **API Documentation**: `http://localhost:8000/docs`
- **Integration Test**: `http://localhost:3000/test`

## ✅ **What Was Cleaned Up**

1. **Removed Duplicate Directory**: Deleted `bluecart-erp-backend/` nested directory
2. **Consolidated Files**: Moved important files from nested directory to root
3. **Single Environment**: Using one `.env` file in the root backend directory
4. **Simplified Structure**: Clean, single-level backend organization

## 🔧 **Available Server Options**

Choose which backend server to run based on your needs:

### Simple Server (Currently Running)
```bash
python main_simple.py
```
- In-memory storage
- No database required
- Perfect for development/testing

### Advanced Server (with Database)
```bash
python main_fastapi.py
```
- PostgreSQL integration
- Full CRUD operations
- Production-ready features

### PostgreSQL Server
```bash
python main_postgres.py
```
- Direct PostgreSQL integration
- Advanced database features
- Requires database setup

## 🎯 **Integration Status**

✅ Frontend and Backend integrated successfully
✅ CORS properly configured
✅ API client working
✅ Test page functional
✅ Clean, single backend directory