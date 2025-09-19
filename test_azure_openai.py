"""
Simple Azure OpenAI test script to isolate the API connectivity issue
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Use the exact same configuration as the main service
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "model-router")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

print("Testing Azure OpenAI Configuration:")
print("Endpoint: {AZURE_OPENAI_ENDPOINT}")
print("Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")
print("API Version: {AZURE_OPENAI_API_VERSION}")
print("API Key: {'***' + str(AZURE_OPENAI_API_KEY)[-4:] if AZURE_OPENAI_API_KEY else 'NOT SET'}")

try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION
    )

    # Simple test call
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=10
    )

    print("\n✅ SUCCESS: Azure OpenAI API is working!")
    print("Response: {response.choices[0].message.content}")

except Exception as e:
    print("\n❌ ERROR: Azure OpenAI API failed")
    print("Error details: {str(e)}")
    print("Error type: {type(e).__name__}")
