"""
Test markers for comprehensive testing pipeline
"""

import pytest

# Test categories
unit = pytest.mark.unit
integration = pytest.mark.integration
performance = pytest.mark.performance
ai = pytest.mark.ai
critical = pytest.mark.critical
regression = pytest.mark.regression

# Timeout markers
quick = pytest.mark.timeout(30)
medium = pytest.mark.timeout(120) 
long = pytest.mark.timeout(600)

# Component markers
database = pytest.mark.database
api = pytest.mark.api
websocket = pytest.mark.websocket
autonomous = pytest.mark.autonomous
security = pytest.mark.security
