# 🚀 RAI Compliance Backend - Deployment Guide

## ✅ Pre-Deployment Checklist
- [x] Code committed and pushed to GitHub
- [x] render.yaml configuration file ready
- [x] requirements.txt includes all dependencies
- [x] Application syntax validated
- [x] Health check endpoint available at `/api/v1/health`

## 🌐 Render Deployment Instructions

### Step 1: Access Render Dashboard
1. Go to [render.com](https://render.com)
2. Sign in with your GitHub account
3. Click "New +" and select "Web Service"

### Step 2: Connect Repository
1. Select "Build and deploy from a Git repository"
2. Connect your GitHub account if not already connected
3. Find and select repository: `vithaluntold/rai-compliance-backend`
4. Select branch: `master`

### Step 3: Configure Service Settings
```yaml
Name: rai-compliance-backend
Region: Oregon (US West) - or your preferred region
Branch: master
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 4: Set Environment Variables
⚠️ **CRITICAL**: Add these environment variables in Render dashboard:

**Required Variables:**
```
AZURE_OPENAI_API_KEY=your_actual_api_key_here
AZURE_OPENAI_ENDPOINT=your_actual_endpoint_here
AZURE_OPENAI_DEPLOYMENT_NAME=model-router
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
```

**Optional Environment Variables:**
```
PYTHON_VERSION=3.11
PORT=10000
```

### Step 5: Deploy
1. Review all settings
2. Click "Create Web Service"
3. Wait for deployment to complete (usually 5-10 minutes)

## 🔧 Post-Deployment Verification

### Test Endpoints
Once deployed, test these URLs (replace with your actual Render URL):

1. **Health Check:**
   ```
   GET https://your-app-name.onrender.com/api/v1/health
   ```

2. **API Documentation:**
   ```
   GET https://your-app-name.onrender.com/docs
   ```

3. **Root Endpoint:**
   ```
   GET https://your-app-name.onrender.com/
   ```

### Expected Responses
- Health check should return: `{"status": "healthy"}`
- Docs should show FastAPI interactive documentation
- Root should return application info

## 🚨 Troubleshooting

### Common Issues:

1. **Build Failed - Missing Dependencies**
   - Check requirements.txt has all packages
   - Verify Python version compatibility

2. **Start Failed - Port Issues**
   - Ensure start command uses `--port $PORT`
   - Don't hardcode port numbers

3. **Runtime Errors - Environment Variables**
   - Verify all Azure OpenAI credentials are set
   - Check variable names match exactly

4. **500 Errors - Application Issues**
   - Check logs in Render dashboard
   - Verify file paths and imports

### Render-Specific Considerations:
- Free tier has 512MB RAM limit
- Services sleep after 15 minutes of inactivity
- Upgrade to paid plan for production workloads

## 📊 Monitoring

### Render Dashboard Features:
- Real-time logs
- Metrics and performance data
- Deploy history
- Environment variable management

### Application Monitoring:
- Monitor `/api/v1/health` endpoint
- Set up external monitoring (e.g., UptimeRobot)
- Check Azure OpenAI usage quotas

## 🔄 Updates and Redeployment

### Automatic Deployment:
- Render auto-deploys on new commits to master branch
- Monitor deploy logs for any issues

### Manual Deployment:
1. Go to your service in Render dashboard
2. Click "Manual Deploy" 
3. Select "Latest commit"

## 🎯 Production Readiness

### Current Status: ✅ READY FOR DEPLOYMENT
- All syntax checks passed
- Dependencies verified
- Configuration validated
- Health endpoints available

### Recommended Next Steps:
1. Deploy to Render staging environment first
2. Test all API endpoints thoroughly
3. Monitor performance and logs
4. Set up domain name (optional)
5. Configure CDN if needed (optional)

---

**Repository:** https://github.com/vithaluntold/rai-compliance-backend
**Deployment Platform:** Render.com
**Status:** Ready for production deployment

For issues, check Render logs or application health endpoints.