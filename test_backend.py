import requests
import json

def test_connection():
    try:
        # Test / (index)
        resp = requests.get("http://localhost:8000/")
        print(f"Index status: {resp.status_code}")
        
        # Test /upload with a dummy file
        files = {'file': ('test.txt', 'This is a test document about artificial intelligence.')}
        resp = requests.post("http://localhost:8000/upload", files=files)
        print(f"Upload status: {resp.status_code}")
        print(f"Upload response: {resp.text}")
        
        if resp.status_code == 200:
            # Test /ask
            data = {"question": "What is this document about?"}
            resp = requests.post("http://localhost:8000/ask", json=data)
            print(f"Ask status: {resp.status_code}")
            print(f"Ask response: {resp.text}")
            
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    test_connection()
