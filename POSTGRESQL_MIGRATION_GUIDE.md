# 🐘 PostgreSQL Configuration Guide for Render

## 📋 **Setup Instructions**

You mentioned you have PostgreSQL on Render. Here's how to configure your BlueCart ERP backend to use it:

### **Step 1: Get Your Render PostgreSQL Credentials**

1. Go to your Render Dashboard
2. Navigate to your PostgreSQL service
3. Copy the **Internal Database URL** (it should look like this):
   ```
   postgresql://username:password@hostname:5432/database_name
   ```

### **Step 2: Update Environment Variables**

Update your `.env` file with your actual Render PostgreSQL credentials:

```bash
# Replace with your actual Render PostgreSQL URL
DATABASE_URL=postgresql://your_username:your_password@your_host:5432/your_database

# Application Settings
SECRET_KEY=your-secret-key-here-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production

# CORS Origins (add your frontend URLs)
CORS_ORIGINS=["http://localhost:3000", "https://your-frontend-url.onrender.com"]
```

### **Step 3: Install Required Dependencies**

Make sure you have the PostgreSQL adapter installed:

```bash
pip install pg8000  # Already in requirements.txt
# OR
pip install asyncpg  # For async version
```

### **Step 4: Switch to PostgreSQL Backend**

I've created two PostgreSQL-enabled backends for you:

#### **Option A: Synchronous Backend (Recommended for simplicity)**
```bash
# Stop the current in-memory backend
# Start the PostgreSQL backend
python main_fastapi.py
```

#### **Option B: Async Backend (Better performance)**
```bash
# Use the async version
python main_postgres.py
```

### **Step 5: Database Migration**

The backend will automatically create the required tables when it starts:
- `shipments` - Store all shipment data
- `hubs` - Store hub information  
- `users` - Store user accounts
- `routes` - Store delivery routes

### **Step 6: Test the Connection**

1. Start the backend:
   ```bash
   python main_fastapi.py  # or main_postgres.py
   ```

2. Check the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

3. You should see:
   ```json
   {
     "status": "healthy",
     "database": "connected",
     "timestamp": "2025-10-08T..."
   }
   ```

## 🔧 **Configuration Files Updated**

### **Updated `.env` file:**
```bash
DATABASE_URL=postgresql://your_render_credentials_here
```

### **Available Backend Options:**

1. **`main_simple.py`** - ❌ In-memory storage (current)
2. **`main_fastapi.py`** - ✅ PostgreSQL with pg8000 
3. **`main_postgres.py`** - ✅ PostgreSQL with asyncpg

## 🚀 **Migration Steps**

### **From In-Memory to PostgreSQL:**

1. **Stop current backend:**
   ```bash
   # Press Ctrl+C in the terminal running main_simple.py
   ```

2. **Update environment:**
   ```bash
   # Edit .env file with your Render PostgreSQL URL
   DATABASE_URL=postgresql://your_credentials
   ```

3. **Start PostgreSQL backend:**
   ```bash
   python main_fastapi.py
   ```

4. **Verify connection:**
   ```bash
   # Check the logs for "✅ Database tables created successfully"
   # Test API: curl http://localhost:8000/docs
   ```

## 📊 **Database Schema**

The backend will create these tables automatically:

### **Shipments Table:**
```sql
CREATE TABLE shipments (
    id VARCHAR(50) PRIMARY KEY,
    tracking_number VARCHAR(100) UNIQUE NOT NULL,
    sender_name VARCHAR(255) NOT NULL,
    sender_address TEXT NOT NULL,
    receiver_name VARCHAR(255) NOT NULL,
    receiver_address TEXT NOT NULL,
    package_details TEXT NOT NULL,
    weight DECIMAL(10,2) NOT NULL,
    dimensions JSONB NOT NULL,
    service_type VARCHAR(50) DEFAULT 'standard',
    cost DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    events JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Hubs Table:**
```sql
CREATE TABLE hubs (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    address TEXT NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    capacity INTEGER NOT NULL,
    current_load INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🎯 **Next Steps**

1. **Get your Render PostgreSQL credentials**
2. **Update the .env file** with your DATABASE_URL
3. **Stop the current backend** (main_simple.py)
4. **Start the PostgreSQL backend** (main_fastapi.py)
5. **Test the connection** via /health endpoint
6. **Verify data persistence** by creating a shipment and restarting the server

## ⚠️ **Important Notes**

- **Backup First**: If you have test data in the current in-memory system, it will be lost when switching
- **Environment Variables**: Make sure your `.env` file is properly configured
- **Network Access**: Ensure your local environment can connect to Render's PostgreSQL
- **SSL Requirements**: Render PostgreSQL might require SSL connections

## 🆘 **Troubleshooting**

### **Connection Errors:**
```
❌ Database connection error: connection refused
```
**Solution**: Check your DATABASE_URL and network connectivity

### **Authentication Errors:**
```
❌ Database connection error: authentication failed
```
**Solution**: Verify your username/password in the DATABASE_URL

### **SSL Errors:**
```
❌ Database connection error: SSL required
```
**Solution**: Add `?sslmode=require` to your DATABASE_URL

---

**Ready to migrate?** Just provide your Render PostgreSQL credentials and I'll help you complete the migration!