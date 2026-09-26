import requests
import time
import json

def test_schedule_api():
    """Test the schedule API endpoints"""
    base_url = "http://localhost:5000"
    
    print("=== Testing Schedule API ===")
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test 2: Check initial status
    print("\n2. Checking initial status...")
    try:
        response = requests.get(f"{base_url}/schedule/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Start the schedule
    print("\n3. Starting schedule...")
    try:
        response = requests.post(f"{base_url}/schedule")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Schedule started successfully!")
        else:
            print("❌ Failed to start schedule")
            return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test 4: Check status after starting
    print("\n4. Checking status after starting...")
    try:
        response = requests.get(f"{base_url}/schedule/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Wait a bit and check status again
    print("\n5. Waiting 10 seconds to see logs...")
    time.sleep(10)
    
    print("\n6. Checking status again...")
    try:
        response = requests.get(f"{base_url}/schedule/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Stop the schedule
    # print("\n7. Stopping schedule...")
    # try:
    #     response = requests.post(f"{base_url}/schedule/stop")
    #     print(f"Status: {response.status_code}")
    #     print(f"Response: {json.dumps(response.json(), indent=2)}")
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # Test 7: Final status check
    print("\n8. Final status check...")
    try:
        response = requests.get(f"{base_url}/schedule/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_schedule_api()
