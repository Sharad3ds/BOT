import requests
import json
import time

def test_live_app():
    """Test your live Render app"""
    
    # Replace with your actual Render URL
    # Your app URL will be: https://your-app-name.onrender.com
    # Check your Render dashboard for the exact URL
    base_url = "https://smbot-v2.onrender.com"  # Replace with your actual URL
    #base_url = "http://127.0.0.1:5000"
    print("=== Testing Live Render App ===")
    print(f"Base URL: {base_url}")
    
    # # Test 1: Health check
    # print("\n1. Testing health check...")
    # try:
    #     response = requests.get(f"{base_url}/health", timeout=10)
    #     print(f"Status: {response.status_code}")
    #     print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return
    
    # Test 2: Start schedule
    print("\n2. Starting schedule...")
    try:
        response = requests.post(f"{base_url}/schedule", timeout=50)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # # Test 3: Check status
    # print("\n3. Checking status...")
    # try:
    #     response = requests.get(f"{base_url}/schedule/status", timeout=10)
    #     print(f"Status: {response.status_code}")
    #     print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # # Test 4: Wait and check again
    # print("\n4. Waiting 5 seconds...")
    # time.sleep(5)
    
    # print("\n5. Checking status again...")
    # try:
    #     response = requests.get(f"{base_url}/schedule/status", timeout=10)
    #     print(f"Status: {response.status_code}")
    #     print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # # Test 5: Stop schedule
    # print("\n6. Stopping schedule...")
    # try:
    #     response = requests.post(f"{base_url}/schedule/stop", timeout=10)
    #     print(f"Status: {response.status_code}")
    #     print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_live_app()
