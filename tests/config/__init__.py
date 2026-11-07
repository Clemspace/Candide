"""
Config System Tests
===================

Comprehensive test suite for the Candide configuration system.

Test Structure:
- test_config.py: Core config classes and functionality
- test_loaders.py: Config loading and file handling
- test_validators.py: Validation and compatibility checks
- test_integration.py: End-to-end integration tests

Running Tests:
    # Run all config tests
    pytest tests/config/

    # Run specific test file
    pytest tests/config/test_config.py -v
    
    # Run specific test class
    pytest tests/config/test_config.py::TestModelConfig -v
    
    # Run with coverage
    pytest tests/config/ --cov=ramanujan.config --cov-report=html
"""

__all__ = []