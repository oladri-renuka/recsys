from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from inference import SASRecInference
from load_movies import load_movie_titles
from pathlib import Path

app = FastAPI(
    title="Movie Recommendation API",
    description="Sequential recommendation engine using SASRec",
    version="1.0.0"
)

# Load model and movies at startup
recommender = SASRecInference(device="cpu")
movie_titles = load_movie_titles()

class RecommendationRequest(BaseModel):
    user_id: int
    watched_movies: List[int]
    num_recommendations: int = 10
    exclude_history: bool = True

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    watched_count: int
    recommendations: List[MovieRecommendation]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "SASRec",
        "device": str(recommender.device),
        "movies_loaded": len(movie_titles)
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a user.
    
    Parameters:
    - user_id: Unique user identifier
    - watched_movies: List of movie IDs user has watched (1-3416)
    - num_recommendations: Number of recommendations to return (1-20)
    - exclude_history: Whether to exclude already watched movies
    
    Returns:
    - user_id: Echo of input user ID
    - watched_count: Number of watched movies
    - recommendations: List of [movie_id, title, score] objects
    """
    
    try:
        if len(request.watched_movies) == 0:
            raise HTTPException(status_code=400, detail="watched_movies cannot be empty")
        
        if len(request.watched_movies) > 200:
            request.watched_movies = request.watched_movies[-200:]
        
        invalid = [x for x in request.watched_movies if x < 1 or x > 3416]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid movie IDs: {invalid}")
        
        recommendations = recommender.recommend(
            request.watched_movies,
            request.num_recommendations,
            request.exclude_history
        )
        
        return RecommendationResponse(
            user_id=request.user_id,
            watched_count=len(request.watched_movies),
            recommendations=[
                MovieRecommendation(
                    movie_id=item_id,
                    title=movie_titles.get(item_id, f"Movie {item_id}"),
                    score=float(score)
                )
                for item_id, score in recommendations
            ]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/batch-recommend")
async def batch_recommend(user_ids: str, watched_movies_json: str):
    """
    Get recommendations for multiple users.
    
    Parameters:
    - user_ids: Comma-separated user IDs (1,2,3)
    - watched_movies_json: JSON string of user histories
    
    Example:
    /batch-recommend?user_ids=1,2&watched_movies_json={"1":[5,10,15],"2":[20,25,30]}
    """
    try:
        import json
        user_ids_list = [int(x.strip()) for x in user_ids.split(",")]
        histories = json.loads(watched_movies_json)
        
        results = []
        for user_id in user_ids_list:
            if str(user_id) not in histories:
                results.append({
                    "user_id": user_id,
                    "error": "No watch history provided"
                })
                continue
            
            history = histories[str(user_id)]
            recs = recommender.recommend(history, n_recommendations=10)
            
            results.append({
                "user_id": user_id,
                "recommendations": [
                    {
                        "movie_id": mid,
                        "title": movie_titles.get(mid, f"Movie {mid}"),
                        "score": float(s)
                    }
                    for mid, s in recs
                ]
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/movie/{movie_id}")
async def get_movie(movie_id: int):
    """Get a specific movie by ID."""
    if movie_id not in movie_titles:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    
    return {
        "movie_id": movie_id,
        "title": movie_titles[movie_id]
    }

@app.get("/search")
async def search_movies(query: str, limit: int = 10):
    """Search movies by name."""
    query_lower = query.lower()
    results = [
        {"movie_id": mid, "title": title}
        for mid, title in movie_titles.items()
        if query_lower in title.lower()
    ][:limit]
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

@app.get("/model-info")
async def model_info():
    """Get model information."""
    return {
        "model_name": "SASRec",
        "parameters": 211950,
        "hidden_dim": 50,
        "num_blocks": 2,
        "max_sequence_length": 200,
        "num_items": 3416,
        "num_users_trained": 6040,
        "test_hit_at_10": 0.7849,
        "test_ndcg_at_10": 0.5811,
        "inference_latency_ms": 2.33,
        "device": str(recommender.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)