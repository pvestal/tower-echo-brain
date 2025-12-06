#!/usr/bin/env python3
"""
Tests for OPM HR Monitor
"""

import pytest
import asyncio
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitors.opm_hr_monitor import OPMHRMonitor


@pytest.mark.asyncio
async def test_opm_monitor_initialization():
    """Test OPM monitor initializes correctly"""
    monitor = OPMHRMonitor()
    assert monitor.opm_base_url == "https://www.opm.gov"
    assert isinstance(monitor.training_modules, list)
    assert isinstance(monitor.last_check, dict)


@pytest.mark.asyncio
async def test_training_compliance_check():
    """Test training compliance checking"""
    monitor = OPMHRMonitor()
    compliance = await monitor.check_training_compliance()

    assert 'total_modules' in compliance
    assert 'active_modules' in compliance
    assert 'outdated_modules' in compliance
    assert 'missing_modules' in compliance
    assert 'recommendations' in compliance
    assert isinstance(compliance['recommendations'], list)


@pytest.mark.asyncio
async def test_monitoring_cycle():
    """Test full monitoring cycle"""
    monitor = OPMHRMonitor()

    # Mock to avoid actual web requests
    async def mock_opm_updates():
        return []

    async def mock_cfr_updates():
        return []

    monitor.check_opm_updates = mock_opm_updates
    monitor.check_5cfr_updates = mock_cfr_updates

    results = await monitor.run_monitoring_cycle()

    assert 'timestamp' in results
    assert 'opm_updates' in results
    assert 'cfr_updates' in results
    assert 'training_compliance' in results
    assert 'actions_needed' in results
    assert isinstance(results['actions_needed'], list)


@pytest.mark.asyncio
async def test_compliance_report_generation():
    """Test compliance report generation"""
    monitor = OPMHRMonitor()

    # Mock to avoid actual web requests
    async def mock_cycle():
        return {
            'timestamp': datetime.now().isoformat(),
            'opm_updates': [],
            'cfr_updates': [],
            'training_compliance': {
                'total_modules': 4,
                'active_modules': 4,
                'outdated_modules': 0,
                'missing_modules': []
            },
            'actions_needed': []
        }

    monitor.run_monitoring_cycle = mock_cycle
    report = await monitor.generate_compliance_report()

    assert isinstance(report, str)
    assert "Federal HR Compliance Report" in report
    assert "OPM Updates" in report
    assert "Training Compliance" in report