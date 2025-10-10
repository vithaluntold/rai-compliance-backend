# PostgreSQL Setup on Render.com

## 🎯 **Quick Overview**
This guide sets up a PostgreSQL database on Render for persistent document storage in your RAI Compliance Tool.

## 📋 **Prerequisites**
- Render.com account (free tier available)
- Your backend code ready for deployment
- PostgreSQL driver already added to requirements.txt ✅

## 🔧 **Step 1: Create PostgreSQL Database**

### 1.1 Login to Render Dashboard
1. Go to https://render.com
2. Sign in with your account
3. Click "New +" button in top right

### 1.2 Create Database Service
1. Select **"PostgreSQL"** from the service types
2. Fill in database details:
   ```
   Name: rai-compliance-database
   Database: rai_compliance_db
   User: rai_admin
   Region: Oregon (US West) or your preferred region
   PostgreSQL Version: 15 (recommended)
   Plan: Free ($0/month) or Starter ($7/month for better performance)
   ```
3. Click **"Create Database"**

### 1.3 Wait for Database Creation
- Database creation takes 2-5 minutes
- Status will change from "Creating" to "Available"
- ✅ **Green status = Ready to use**

## 🔗 **Step 2: Get Database Connection Details**

### 2.1 Access Database Dashboard
1. Click on your database name in Render dashboard
2. Go to "Connect" tab
3. You'll see connection details like:

```
Host: dpg-xxxxxxx-a.oregon-postgres.render.com
Port: 5432
Database: rai_compliance_db
Username: rai_admin
Password: [auto-generated secure password]
```

### 2.2 Copy Connection String
Render provides a full connection string:
```
postgresql://rai_admin:password123@dpg-xxxxxxx-a.oregon-postgres.render.com:5432/rai_compliance_db
```

## ⚙️ **Step 3: Configure Backend Environment**

### 3.1 Update render.yaml
```yaml
services:
  - type: web
    name: rai-compliance-backend
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python main.py"
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: rai-compliance-database
          property: connectionString
      - key: ENVIRONMENT
        value: production
      - key: PORT
        value: 10000
```

### 3.2 Environment Variables Setup
In your Render web service settings:
```
DATABASE_URL = postgresql://rai_admin:password@host:5432/rai_compliance_db
ENVIRONMENT = production
PORT = 10000
```

## 🔄 **Step 4: Deploy with Database**

### 4.1 Your Code is Ready ✅
Your `persistent_storage_enhanced.py` already supports PostgreSQL:
- Auto-detects DATABASE_URL environment variable
- Creates tables automatically on first run
- Handles both SQLite (dev) and PostgreSQL (production)

### 4.2 Deploy Backend
1. Connect your GitHub repo to Render
2. Select the backend folder: `/render-backend`
3. Deploy will automatically:
   - Install dependencies (including psycopg2-binary)
   - Connect to PostgreSQL
   - Create database tables
   - Start your FastAPI server

### 4.3 Verify Database Connection
After deployment, check logs for:
```
INFO: Database connected: PostgreSQL
INFO: Tables created successfully
INFO: Starting application on port 10000
```

## 🧪 **Step 5: Test Database Functionality**

### 5.1 Health Check
```bash
curl https://your-backend-url.onrender.com/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "PostgreSQL",
  "tables": ["files", "analysis_results", "processing_locks"]
}
```

### 5.2 Test Document Storage
Upload a document via your frontend and verify it persists in PostgreSQL.

## 📊 **Database Schema Created Automatically**

Your app will create these tables:

### `files` table:
```sql
- id (TEXT PRIMARY KEY)
- filename (TEXT)
- content (TEXT)  -- Full file content in PostgreSQL
- upload_time (TIMESTAMP)
- file_size (INTEGER)
- file_type (TEXT)
```

### `analysis_results` table:
```sql
- document_id (TEXT PRIMARY KEY)
- status (TEXT)
- metadata (JSONB)  -- Rich JSON support in PostgreSQL
- sections (JSONB)
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
```

### `processing_locks` table:
```sql
- document_id (TEXT PRIMARY KEY)
- status (TEXT)
- created_at (TIMESTAMP)
```

## 🎯 **Benefits of PostgreSQL on Render**

### ✅ **Production Features:**
- **Persistent Storage**: Data survives container restarts
- **JSONB Support**: Efficient metadata and section storage
- **Concurrent Access**: Multiple users can upload simultaneously
- **Automatic Backups**: Render handles daily backups
- **SSL Connections**: Secure database access
- **Monitoring**: Built-in database metrics

### ✅ **Performance:**
- **Indexed Queries**: Fast document retrieval
- **Connection Pooling**: Efficient database connections
- **Memory Optimization**: Better than file-based storage

## 🚀 **What Happens After Setup**

1. **Document Upload** → Stored in PostgreSQL `files` table
2. **Analysis Processing** → Results in `analysis_results` table  
3. **User Navigation** → Fast retrieval from database
4. **No File System Dependency** → Works across container restarts
5. **Multi-User Support** → Each document has unique database entry

## 🔧 **Troubleshooting**

### Connection Issues:
```bash
# Check if DATABASE_URL is set correctly
echo $DATABASE_URL

# Test connection manually
python -c "import psycopg2; conn=psycopg2.connect('your-database-url'); print('Connected!')"
```

### Database Not Found:
- Verify database name matches in render.yaml
- Check environment variables are linked correctly
- Ensure database status is "Available" in Render dashboard

## 💡 **Cost Optimization**

### Free Tier Limits:
- **Database**: 1GB storage, 97 hours/month runtime
- **Web Service**: 750 hours/month
- **Perfect for development and testing**

### Paid Tier Benefits ($7/month database):
- **Always-on database** (no sleep mode)
- **10GB+ storage**
- **Better performance for production**

---

**🎉 Your PostgreSQL setup will eliminate all localhost dependencies and provide true cloud persistence!**