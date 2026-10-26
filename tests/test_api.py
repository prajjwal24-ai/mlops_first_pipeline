# Import TestClient so we cans send simulated requests HTTP Requests
from fastapi.testclient import TestClient

# Import application object defined in main.py
from main import app

# Reusable test client for all test functions
client = TestClient(app)

def test_health_endpoint():
    """
    Verify that the application is running and can respond to a health-check request
    """

    response = client.get("/health")

    # Used to raise Assertion Errors if the condition fails
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_valid_iris_prediction():
    """
    Verify that we are getting valid prediction
    """

    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    # Send a simulated POST request with JSON input
    response = client.post("/predict", json = payload)

    # Endpoint should accept the valid input
    assert response.status_code == 200

    result = response.json()
    
    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in ['setosa', 'versicolor', 'virginica']
    assert result["prediction"] in [0, 1, 2]