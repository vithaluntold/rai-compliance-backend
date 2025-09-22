@echo off
echo 🚀 RAI Compliance Platform - User Flow Test Runner
echo ====================================================
echo.
echo This script will run comprehensive user flow tests to identify disconnects.
echo Make sure both servers are running before proceeding:
echo.
echo ✅ Backend: http://localhost:8000 (python main.py)
echo ✅ Frontend: http://localhost:3000 (npm run dev)
echo.
pause
echo.
echo 🔍 Step 1: Testing server connectivity...
python test_connectivity.py
echo.
echo 🔍 Step 2: Running simple user flow test...
python test_simple_user_flow.py
echo.
echo 🔍 Step 3: Running comprehensive backend API test...
python test_user_flow_comprehensive.py
echo.
echo ✅ User flow testing complete!
echo Check the output above for any disconnects or issues.
echo.
pause