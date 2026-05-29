import requests
import json

BASE_URL = "http://localhost:8000"

print("="*60)
print("TESTING FASTAPI SERVER")
print("="*60)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
data = response.json()
print(f"Status: {data['status']}, Movies: {data['movies_loaded']}")

# Test 2: Single user recommendation WITH TITLES
print("\n2. Testing /recommend endpoint (with titles)...")
payload = {
    "user_id": 1,
    "watched_movies": [1, 50, 100],
    "num_recommendations": 5
}
response = requests.post(f"{BASE_URL}/recommend", json=payload)
print(f"Status: {response.status_code}")
data = response.json()
print(f"User ID: {data['user_id']}")
print(f"Watched: {data['watched_count']} movies")
print(f"Recommendations:")
for i, rec in enumerate(data['recommendations'], 1):
    print(f"  {i}. {rec['title']:<40} (Score: {rec['score']:.3f})")

# Test 3: Search movies
print("\n3. Testing /search endpoint...")
response = requests.get(f"{BASE_URL}/search?query=Matrix&limit=3")
print(f"Status: {response.status_code}")
data = response.json()
print(f"Found {data['count']} movies matching '{data['query']}':")
for movie in data['results']:
    print(f"  • {movie['title']}")

# Test 4: Get single movie
print("\n4. Testing /movie endpoint...")
response = requests.get(f"{BASE_URL}/movie/1")
print(f"Status: {response.status_code}")
data = response.json()
print(f"Movie {data['movie_id']}: {data['title']}")

print("\n" + "="*60)
print("ALL TESTS PASSED")
print("="*60)