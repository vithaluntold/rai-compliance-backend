# Render Deployment Fix Instructions

## Problem
Render is trying to deploy from `vithaluntold/rai-compliance-backend` but our backend code is actually in `vithaluntold/rai-compliance-frontend` repository.

## Solution
Update your Render service configuration:

1. **Go to your Render Dashboard**
2. **Find your `rai-compliance-backend` service**
3. **Click on "Settings"**
4. **Update Repository Settings:**
   - Repository: `vithaluntold/rai-compliance-frontend`
   - Branch: `master`
   - Root Directory: `Audricc all/render-backend` (or whatever the correct path is)

5. **Update Build Settings:**
   - Build Command: `cd "Audricc all/render-backend" && pip install --no-cache-dir -r requirements.txt`
   - Start Command: `cd "Audricc all/render-backend" && uvicorn main:app --host 0.0.0.0 --port $PORT`

6. **Save and Deploy**

## Alternative: Push Backend Code to Correct Repository

If you prefer to use the separate backend repository:

```bash
# Add the backend repository as a remote
git remote add backend https://github.com/vithaluntold/rai-compliance-backend.git

# Push your backend code to the backend repository
git subtree push --prefix="Audricc all/render-backend" backend master
```

## Current Status
- Local repository: `rai-compliance-frontend`
- Render expecting: `rai-compliance-backend`
- Backend code location: `Audricc all/render-backend/` subdirectory