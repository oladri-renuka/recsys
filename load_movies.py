from pathlib import Path
import pandas as pd

def load_movie_titles(ml_1m_path="ml-1m"):
    """Load MovieLens movie titles."""
    movies_path = Path(ml_1m_path) / "movies.dat"
    
    if not movies_path.exists():
        print(f"Warning: {movies_path} not found. Using generic titles.")
        return {i: f"Movie {i}" for i in range(1, 3417)}
    
    # FIX: Use latin-1 encoding (MovieLens uses this)
    movies = pd.read_csv(
        movies_path,
        sep="::",
        header=None,
        names=['item_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'  # ADD THIS LINE
    )
    
    return dict(zip(movies['item_id'], movies['title']))

if __name__ == "__main__":
    titles = load_movie_titles()
    print(f"Loaded {len(titles)} movies")
    print(f"Sample: Movie 1 = {titles.get(1, 'Unknown')}")