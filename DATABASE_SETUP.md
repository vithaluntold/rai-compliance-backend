# Render PostgreSQL Database Configuration

## Steps to Setup:

### 1. Create PostgreSQL Database on Render
```
1. Go to Render Dashboard
2. Click "New +" → "PostgreSQL"
3. Name: rai-compliance-db
4. Region: Choose closest to users
5. Plan: Free tier (suitable for testing)
```

### 2. Database Connection Details
```
Database URL will be provided by Render:
postgresql://username:password@host:port/database_name

Environment Variables to set in Render:
- DATABASE_URL: (provided by Render PostgreSQL)
- POSTGRES_HOST: (from Render)
- POSTGRES_PORT: (from Render)
- POSTGRES_DB: (from Render)
- POSTGRES_USER: (from Render)
- POSTGRES_PASSWORD: (from Render)
```

### 3. Benefits over SQLite
- ✅ True persistence across deployments
- ✅ Better performance for concurrent users
- ✅ ACID compliance
- ✅ Automatic backups
- ✅ Scales with application growth