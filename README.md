# SASRec Movie Recommendation API

A production-ready sequential recommendation engine using self-attention, deployed on AWS EC2.

## Overview

This is a complete implementation of **SASRec** (Self-Attentive Sequential Recommendation) trained on MovieLens-1M and deployed as a REST API on AWS.

- **Model**: SASRec (211,950 parameters)
- **Dataset**: MovieLens-1M (6,040 users, 3,416 movies, 999,611 interactions)
- **Accuracy**: NDCG@10 58.11% 
- **Latency**: 2.3ms per recommendation (CPU)
- **Deployment**: AWS EC2 (t3.micro, free tier)

## Live API

**Base URL**: `http://18.225.169.201:8000`

## Quick Start

### Test Health

```bash
curl http://18.225.169.201:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "SASRec",
  "device": "cpu",
  "movies_loaded": 3883
}
```

### Get Recommendations

```bash
curl -X POST "http://18.225.169.201:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "watched_movies": [1, 50, 100],
    "num_recommendations": 5
  }'
```

Response:
```json
{
  "user_id": 1,
  "watched_count": 3,
  "recommendations": [
    {
      "movie_id": 333,
      "title": "Tommy Boy (1995)",
      "score": 3.52
    },
    {
      "movie_id": 1011,
      "title": "Herbie Rides Again (1974)",
      "score": 3.07
    }
  ]
}
```

### Search Movies

```bash
curl "http://18.225.169.201:8000/search?query=Matrix"
```

Response:
```json
{
  "query": "Matrix",
  "results": [
    {"movie_id": 1500, "title": "The Matrix (1999)"}
  ],
  "count": 1
}
```

### Interactive API Documentation

```
http://18.225.169.201:8000/docs
```

(Swagger UI - test all endpoints in browser)

## API Endpoints

### POST /recommend
Get personalized recommendations for a user.

**Request:**
```json
{
  "user_id": 1,
  "watched_movies": [1, 50, 100],
  "num_recommendations": 10,
  "exclude_history": true
}
```

**Parameters:**
- `user_id` (int): Unique user ID
- `watched_movies` (list): Movie IDs user has watched (1-3416)
- `num_recommendations` (int): Number of recommendations (1-20, default 10)
- `exclude_history` (bool): Exclude already-watched movies (default true)

**Response:** Recommendations with titles, IDs, and scores

---

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "SASRec",
  "device": "cpu",
  "movies_loaded": 3883
}
```

---

### GET /search
Search movies by name.

**Parameters:**
- `query` (string): Movie name to search
- `limit` (int): Max results (default 10)

**Response:**
```json
{
  "query": "Matrix",
  "results": [
    {"movie_id": 1500, "title": "The Matrix (1999)"}
  ],
  "count": 1
}
```

---

### GET /model-info
Get model metadata.

**Response:**
```json
{
  "model_name": "SASRec",
  "parameters": 211950,
  "test_hit_at_10": 0.7849,
  "test_ndcg_at_10": 0.5811,
  "inference_latency_ms": 2.33,
  "device": "cpu"
}
```

## Model Architecture

```
Input Sequence: [Movie IDs]
    ↓
Embedding Layer (50-dim)
    ↓
Positional Encoding (50-dim)
    ↓
SASRec Block 1: Multi-Head Attention + FFN
    ↓
SASRec Block 2: Multi-Head Attention + FFN
    ↓
Final LayerNorm
    ↓
Output Logits: [batch_size, seq_len, 3416]
```

**Hyperparameters:**
- Hidden dimension: 50
- Number of blocks: 2
- Attention heads: 1
- Dropout: 0.2
- Max sequence length: 200
- Learning rate: 0.001 (Adam)
- LR scheduler: StepLR(step_size=100, gamma=0.5)

## Project Structure

```
recsys/
├── serve.py              # FastAPI production server
├── inference.py          # Model inference wrapper
├── model.py              # SASRec model architecture
├── data.py               # Data preprocessing
├── dataset.py            # PyTorch dataloaders
├── train.py              # Training script
├── ablation.py           # Ablation studies
├── segment.py            # User segment analysis
├── benchmark.py          # Latency benchmarking
├── load_movies.py        # Movie title loading
├── app.py                # Streamlit UI (local)
├── requirements.txt      # Python dependencies
└── results/
    ├── best_model.pt     # Trained model weights
    ├── training_log.json
    ├── ablation_results.json
    └── segment_analysis.json
```

## Performance Metrics

### Accuracy
- **NDCG@10**: 58.11% 
- **Hit@10**: 78.49% 

### Inference Latency
- Single user (batch 1): 1.81ms
- Batch 16 (real-time): 2.33ms
- Batch 128 (batch processing): 15.30ms
- Max throughput: 8,366 requests/sec

### Training
- Total parameters: 211,950
- Best validation NDCG: 0.5811 
- Training time: ~8 hours (T4 GPU)
- Dataset: 999,611 interactions from 6,040 users
- Note: Checkpoint saved based on best validation NDCG, not training loss

## Key Findings

**1. Extended Training Required**
- 500 epochs needed for convergence
- Early stopping at 100 epochs would miss 40% of improvement
- Learning rate scheduling (StepLR) essential

**2. Model Depth & Positional Encoding Essential**
- Removing second attention block: −28% NDCG (0.5044 → 0.3632)
- Removing positional encoding: −24% NDCG (0.5044 → 0.3835)
- PE impact varies by user segment: most beneficial for short-sequence users (<20 items) where positional information is more discriminative than for long-sequence users (>100 items)

**3. Cold-Start Weakness**
- Model performs best on short sequences (<20 items): Hit@10=79.1%
- Model performs worst on long sequences (>100 items): Hit@10=62.9%
- Self-attention struggles with very long dependencies

**4. No Caching Added**
- Each user has unique watch history
- Cache hit rate would be near 0%
- Model is already fast enough (2.3ms)
- Complexity not justified for latency benefit

## Deployment Details

### AWS Infrastructure
- **Compute**: EC2 t3.micro (free tier)
- **Storage**: S3 bucket for model artifact (0.8 MB)
- **Region**: us-east-2 (Ohio)
- **Uptime**: 24/7

### How to SSH
```bash
# SSH access available for maintainer
```

### View Logs
```bash
tail -f ~/recsys/api.log
```

### Restart API
```bash
pkill -f "python serve.py"
nohup python serve.py > api.log 2>&1 &
```

## Development

### Local Setup
```bash
git clone https://github.com/your-username/recsys.git
cd recsys
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Locally
```bash
# Terminal 1 - API
python serve.py

# Terminal 2 - UI (optional)
streamlit run app.py
```

### Train From Scratch
```bash
python train.py
```

## Citation

Based on the paper: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE 8th International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

## License

MIT

