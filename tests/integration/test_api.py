"""
Integration tests for Market Pulse API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Test health check returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_returns_healthy_status(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "agents" in data

    def test_health_check_includes_version(self, client):
        """Test health check includes version."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data


class TestAgentStatusEndpoint:
    """Tests for agent status endpoint."""

    def test_agent_status_returns_200(self, client):
        """Test agent status returns 200."""
        response = client.get("/api/v1/agents/status")
        assert response.status_code == 200

    def test_agent_status_returns_all_agents(self, client):
        """Test agent status returns all agents."""
        response = client.get("/api/v1/agents/status")
        data = response.json()
        assert data["total_agents"] == 5
        assert data["operational_agents"] == 5


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, client):
        """Test root returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self, client):
        """Test root returns API info."""
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


class TestPulseEndpoint:
    """Tests for market pulse endpoint."""

    @pytest.fixture
    def valid_request(self):
        """Create valid request payload."""
        return {
            "user_id": "test_user_123",
            "news_filter": "all",
            "max_news_items": 10,
        }

    def test_pulse_requires_user_id(self, client):
        """Test pulse endpoint requires user_id."""
        response = client.post("/api/v1/pulse", json={})
        assert response.status_code == 422  # Validation error

    def test_pulse_validates_request(self, client, valid_request):
        """Test pulse endpoint validates request."""
        # Remove required field
        del valid_request["user_id"]
        response = client.post("/api/v1/pulse", json=valid_request)
        assert response.status_code == 422
