#!/usr/bin/env python3
"""
Quick deployment helper script for RAI Compliance Backend
This script helps verify deployment readiness and provides next steps.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and return its result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: {e}")
        return False

def check_deployment_readiness():
    """Check if the application is ready for deployment."""
    print("🚀 RAI Compliance Backend - Deployment Readiness Check")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Git status
    total_checks += 1
    if run_command("git status --porcelain", "Git repository is clean"):
        checks_passed += 1
    
    # Check 2: Main files exist
    total_checks += 1
    required_files = ["main.py", "requirements.txt", "render.yaml"]
    if all(Path(f).exists() for f in required_files):
        print("✅ Required deployment files present")
        checks_passed += 1
    else:
        print("❌ Missing required deployment files")
    
    # Check 3: Python syntax
    total_checks += 1
    if run_command("python -m py_compile main.py", "Main application syntax valid"):
        checks_passed += 1
    
    # Check 4: Dependencies installable
    total_checks += 1
    if run_command("pip check", "Dependencies are compatible"):
        checks_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Deployment Readiness: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n🎉 FULLY READY FOR DEPLOYMENT!")
        print("\n📋 Next Steps:")
        print("1. Go to https://render.com")
        print("2. Create new Web Service")
        print("3. Connect to GitHub repository: vithaluntold/rai-compliance-backend")
        print("4. Set environment variables (see DEPLOYMENT_GUIDE.md)")
        print("5. Deploy!")
        
        print("\n🔑 Don't forget to set these environment variables in Render:")
        env_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT", 
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION"
        ]
        for var in env_vars:
            print(f"   - {var}")
            
    else:
        print("\n⚠️  Some checks failed. Please fix issues before deploying.")
    
    return checks_passed == total_checks

def show_render_config():
    """Display the Render configuration."""
    print("\n📄 Render Configuration (render.yaml):")
    try:
        with open("render.yaml", "r") as f:
            content = f.read()
            print(content)
    except FileNotFoundError:
        print("❌ render.yaml not found!")

if __name__ == "__main__":
    if check_deployment_readiness():
        show_render_config()
        
        print("\n🌐 Ready to deploy to Render!")
        print("📖 See DEPLOYMENT_GUIDE.md for detailed instructions.")
    else:
        print("\n🔧 Please fix the issues above before deploying.")
        sys.exit(1)