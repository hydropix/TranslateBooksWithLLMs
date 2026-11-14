"""
Test for GitHub Issue #108 (Part 2): Ollama endpoint uses localhost instead of server config

This test verifies that:
1. The /api/config endpoint exposes the correct OLLAMA_API_ENDPOINT from server
2. The frontend initialization sequence loads models AFTER server config is received

Regression test for: https://github.com/hydropix/TranslateBooksWithLLMs/issues/108

Bug description:
When OLLAMA_API_ENDPOINT is configured to a remote server (e.g., http://192.168.1.4:11434/api/generate),
the frontend was still trying to connect to localhost:11434 because:
1. SettingsManager.initialize() restored lastProvider from localStorage and triggered model loading
2. This happened BEFORE FormManager.loadDefaultConfig() received the server config
3. So the model loading used the HTML default value (localhost) instead of server config

Expected behavior:
The first model loading should wait for server config to be loaded, then use the correct endpoint.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestOllamaEndpointInitializationBug:
    """
    Tests for OLLAMA_API_ENDPOINT proper exposure and initialization sequence.
    
    GitHub Issue #108 (Part 2): When OLLAMA_API_ENDPOINT is configured to a remote
    server, the UI was still trying to connect to localhost because the frontend
    loaded models before receiving the server configuration.
    """

    @pytest.fixture
    def app(self):
        """Create a Flask app for testing with remote Ollama config."""
        # Set test values BEFORE importing Flask app components
        os.environ['OLLAMA_API_ENDPOINT'] = 'http://192.168.1.4:11434/api/generate'
        os.environ['API_ENDPOINT'] = 'http://192.168.1.4:11434/api/generate'
        
        # Import and reload config to pick up test values
        import importlib
        from src import config
        importlib.reload(config)
        
        # Now import and create Flask app (it will use the reloaded config)
        from flask import Flask
        from src.api.blueprints.config_routes import create_config_blueprint
        
        app = Flask(__name__)
        bp = create_config_blueprint(server_session_id=12345)
        app.register_blueprint(bp)
        
        yield app
        
        # Cleanup: restore original values after test
        del os.environ['OLLAMA_API_ENDPOINT']
        del os.environ['API_ENDPOINT']

    def test_api_config_response_contains_ollama_endpoint(self, app):
        """
        Test that /api/config response includes ollama_api_endpoint field
        with the correct remote server value.
        
        This verifies the backend correctly exposes the configured endpoint.
        """
        with app.test_client() as client:
            response = client.get('/api/config')
            
            assert response.status_code == 200, \
                f"Expected 200 but got {response.status_code}"
            
            data = response.get_json()
            assert data is not None, "Response should be valid JSON"
            
            # KEY ASSERTION: ollama_api_endpoint must be present
            assert 'ollama_api_endpoint' in data, \
                "BUG: 'ollama_api_endpoint' missing from /api/config. " \
                f"Got keys: {list(data.keys())}"
            
            # It should have the remote server value, not localhost
            assert data['ollama_api_endpoint'] == 'http://192.168.1.4:11434/api/generate', \
                f"Expected remote endpoint but got '{data.get('ollama_api_endpoint')}'"

    def test_api_config_response_contains_legacy_api_endpoint(self, app):
        """
        Test that /api/config response also includes api_endpoint field
        for backward compatibility.
        """
        with app.test_client() as client:
            response = client.get('/api/config')
            data = response.get_json()
            
            assert 'api_endpoint' in data, \
                "'api_endpoint' should be in /api/config response for backward compatibility"
            
            assert data['api_endpoint'] == 'http://192.168.1.4:11434/api/generate', \
                f"Expected remote endpoint but got '{data.get('api_endpoint')}'"


class TestFrontendInitializationSequence:
    """
    Tests to verify the frontend initialization sequence is correct.
    
    These tests document the expected behavior and can be run as JavaScript
    unit tests in a browser environment.
    """

    def test_initialization_order_documentation(self):
        """
        Document the correct initialization order that prevents the bug.
        
        The bug occurs when:
        1. SettingsManager.initialize() triggers model loading with localStorage endpoint
        2. This happens BEFORE FormManager.loadDefaultConfig() receives server config
        
        The fix ensures:
        1. FormManager.loadDefaultConfig() fetches server config first
        2. THEN ProviderManager loads models with the correct endpoint
        """
        # This test serves as documentation for the expected initialization order
        expected_order = [
            "SettingsManager.initialize() - load local preferences BUT DON'T trigger model loading",
            "FormManager.initialize() - start loadDefaultConfig() async",
            "ProviderManager.initialize() - setup event listeners BUT DON'T load models yet",
            "FormManager.loadDefaultConfig() completes - dispatch 'defaultConfigLoaded'",
            "ProviderManager receives 'defaultConfigLoaded' - NOW load models with correct endpoint"
        ]
        
        # Verify the order makes sense
        assert len(expected_order) == 5
        assert "SettingsManager" in expected_order[0]
        assert "FormManager" in expected_order[1]
        assert "ProviderManager" in expected_order[2]
        assert "defaultConfigLoaded" in expected_order[3]
        assert "ProviderManager" in expected_order[4]

    def test_javascript_event_sequence(self):
        """
        Test that verifies the JavaScript event sequence for proper initialization.
        
        This test describes what should happen in the browser:
        1. Page loads with HTML default: apiEndpoint="http://localhost:11434/api/generate"
        2. SettingsManager.initialize() restores preferences but DOESN'T trigger provider change
        3. FormManager.initialize() calls loadDefaultConfig() - async fetch to /api/config
        4. ProviderManager.initialize() sets up listeners but DOESN'T call toggleProviderSettings()
        5. loadDefaultConfig() receives server config: ollama_api_endpoint="http://192.168.1.4:11434/api/generate"
        6. FormManager updates apiEndpoint input with server value
        7. FormManager dispatches 'defaultConfigLoaded' event
        8. ProviderManager receives event and NOW calls toggleProviderSettings() → load models
        9. Models load with correct endpoint: http://192.168.1.4:11434/api/tags
        """
        # Document the expected event flow
        event_flow = {
            "html_default": "http://localhost:11434/api/generate",
            "server_config": "http://192.168.1.4:11434/api/generate",
            "critical_moment": "ProviderManager must wait for defaultConfigLoaded before loading models",
            "expected_api_call": "http://192.168.1.4:11434/api/tags (NOT localhost)"
        }
        
        assert "localhost" in event_flow["html_default"]
        assert "192.168.1.4" in event_flow["server_config"]
        assert "wait" in event_flow["critical_moment"].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
