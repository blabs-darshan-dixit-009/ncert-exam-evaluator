# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "configuration" in data
    assert "storage" in data


def test_list_available_models():
    """Test listing available models"""
    response = client.get("/models/available")
    assert response.status_code == 200
    data = response.json()
    assert "current_model" in data
    assert "available_models" in data
    assert len(data["available_models"]) > 0


def test_list_trained_models():
    """Test listing trained models"""
    response = client.get("/api/models/list")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total_models" in data


def test_storage_stats():
    """Test storage statistics endpoint"""
    response = client.get("/api/models/storage/stats")
    assert response.status_code == 200
    data = response.json()
    assert "directories" in data
    assert "total_size_mb" in data


def test_training_validation():
    """Test training request validation"""
    # Test with invalid model name (too short)
    response = client.post(
        "/api/training/train",
        json={
            "model_name": "ab",  # Too short
            "training_examples": []
        }
    )
    assert response.status_code == 422  # Validation error
    
    # Test with too few examples
    response = client.post(
        "/api/training/train",
        json={
            "model_name": "test_model",
            "training_examples": [
                {
                    "question": "Test question?",
                    "ideal_answer": "Test answer"
                }
            ]  # Only 1 example, need at least MIN_TRAINING_EXAMPLES
        }
    )
    assert response.status_code in [400, 422]


def test_evaluation_without_model():
    """Test evaluation with non-existent model"""
    response = client.post(
        "/api/evaluation/evaluate",
        json={
            "model_name": "nonexistent_model",
            "question": "Test question?",
            "use_rag": True
        }
    )
    assert response.status_code in [400, 404, 500]


def test_cache_status():
    """Test cache status endpoint"""
    response = client.get("/api/evaluation/cache/status")
    assert response.status_code == 200
    data = response.json()
    assert "cached_models" in data
    assert "num_cached" in data


def test_model_info_not_found():
    """Test getting info for non-existent model"""
    response = client.get("/api/models/info/nonexistent_model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling concurrent requests"""
    import asyncio
    
    async def make_request():
        return client.get("/health")
    
    # Make 5 concurrent requests
    tasks = [make_request() for _ in range(5)]
    responses = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)