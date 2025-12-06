#!/usr/bin/env python3
"""
OPM and Federal HR Training Monitor for Echo Brain
Monitors Office of Personnel Management updates and HR training requirements
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class OPMHRMonitor:
    """Monitor OPM updates and federal HR training requirements"""

    def __init__(self):
        self.opm_base_url = "https://www.opm.gov"
        self.last_check_file = Path("/opt/tower-echo-brain/state/opm_last_check.json")
        self.training_modules = self._load_training_modules()
        self.last_check = self._load_last_check()

    def _load_training_modules(self) -> List[Dict[str, Any]]:
        """Load federal HR training modules"""
        modules = []

        # Check if federal-hr-training directory exists
        hr_training_dir = Path("/opt/tower-echo-brain/federal-hr-training")
        if not hr_training_dir.exists():
            hr_training_dir = Path("./federal-hr-training")

        if hr_training_dir.exists():
            try:
                # Load module information
                for module_dir in hr_training_dir.glob("*/"):
                    if module_dir.is_dir() and module_dir.name.startswith(('01-', '02-', '03-', '04-')):
                        module_info = {
                            'name': module_dir.name,
                            'path': str(module_dir),
                            'last_updated': datetime.fromtimestamp(module_dir.stat().st_mtime),
                            'status': 'active'
                        }
                        modules.append(module_info)

                logger.info(f"✅ Loaded {len(modules)} HR training modules")
            except Exception as e:
                logger.error(f"Failed to load training modules: {e}")

        return modules

    def _load_last_check(self) -> Dict[str, Any]:
        """Load last check status"""
        if self.last_check_file.exists():
            try:
                with open(self.last_check_file) as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            'last_opm_check': None,
            'last_5cfr_check': None,
            'last_training_check': None,
            'known_updates': []
        }

    def _save_last_check(self):
        """Save last check status"""
        self.last_check_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.last_check_file, 'w') as f:
            json.dump(self.last_check, f, indent=2, default=str)

    async def check_opm_updates(self) -> List[Dict[str, Any]]:
        """Check for new OPM policy updates"""
        updates = []

        try:
            # OPM news and updates endpoints
            endpoints = [
                "/news/releases/",
                "/policy-data-oversight/",
                "/retirement/",
                "/healthcare-insurance/",
                "/pay-leave/"
            ]

            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    url = f"{self.opm_base_url}{endpoint}"
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')

                                # Look for updates (simplified parsing)
                                articles = soup.find_all(['article', 'div'], class_=['news-item', 'update', 'announcement'])
                                for article in articles[:5]:  # Latest 5 per category
                                    title_elem = article.find(['h2', 'h3', 'a'])
                                    if title_elem:
                                        update = {
                                            'source': 'OPM',
                                            'category': endpoint.strip('/').split('/')[-1],
                                            'title': title_elem.get_text(strip=True),
                                            'url': url,
                                            'timestamp': datetime.now().isoformat(),
                                            'type': 'policy_update'
                                        }

                                        # Check if this is new
                                        if update['title'] not in [u.get('title') for u in self.last_check.get('known_updates', [])]:
                                            updates.append(update)
                                            logger.info(f"📰 New OPM update: {update['title']}")

                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout checking {url}")
                    except Exception as e:
                        logger.debug(f"Error checking {url}: {e}")

            self.last_check['last_opm_check'] = datetime.now().isoformat()
            if updates:
                self.last_check['known_updates'].extend(updates)
                # Keep only last 100 updates
                self.last_check['known_updates'] = self.last_check['known_updates'][-100:]

            self._save_last_check()

        except Exception as e:
            logger.error(f"Failed to check OPM updates: {e}")

        return updates

    async def check_5cfr_updates(self) -> List[Dict[str, Any]]:
        """Check for 5 CFR (Code of Federal Regulations) updates"""
        updates = []

        try:
            # Check eCFR for Title 5 updates
            ecfr_url = "https://www.ecfr.gov/current/title-5"

            async with aiohttp.ClientSession() as session:
                async with session.get(ecfr_url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Simple check for "Last updated" or revision dates
                        if "Last updated" in html or "effective" in html.lower():
                            update = {
                                'source': '5 CFR',
                                'category': 'regulation',
                                'title': '5 CFR Check - Manual review recommended',
                                'url': ecfr_url,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'regulation_update'
                            }
                            updates.append(update)
                            logger.info("📋 5 CFR check completed - review recommended")

            self.last_check['last_5cfr_check'] = datetime.now().isoformat()
            self._save_last_check()

        except Exception as e:
            logger.error(f"Failed to check 5 CFR updates: {e}")

        return updates

    async def check_training_compliance(self) -> Dict[str, Any]:
        """Check federal HR training compliance status"""
        compliance = {
            'total_modules': len(self.training_modules),
            'active_modules': 0,
            'outdated_modules': 0,
            'missing_modules': [],
            'recommendations': []
        }

        try:
            # Required federal HR training categories
            required_categories = [
                'basic-hr',
                'intermediate-hr',
                'advanced-hr',
                'expert-hr',
                'financial-management',
                'ethics',
                'security-awareness',
                'records-management'
            ]

            found_categories = [m['name'].lower() for m in self.training_modules]

            for category in required_categories:
                if not any(category in fc for fc in found_categories):
                    compliance['missing_modules'].append(category)
                    compliance['recommendations'].append(f"Add {category} training module")

            # Check module age
            for module in self.training_modules:
                if module['status'] == 'active':
                    compliance['active_modules'] += 1

                # Module older than 1 year needs review
                if datetime.now() - module['last_updated'] > timedelta(days=365):
                    compliance['outdated_modules'] += 1
                    compliance['recommendations'].append(f"Review and update {module['name']}")

            self.last_check['last_training_check'] = datetime.now().isoformat()
            self._save_last_check()

            logger.info(f"📚 Training compliance: {compliance['active_modules']}/{compliance['total_modules']} active")

        except Exception as e:
            logger.error(f"Failed to check training compliance: {e}")

        return compliance

    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run complete monitoring cycle"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'opm_updates': [],
            'cfr_updates': [],
            'training_compliance': {},
            'actions_needed': []
        }

        try:
            # Check OPM updates
            logger.info("🔍 Checking OPM updates...")
            results['opm_updates'] = await self.check_opm_updates()

            # Check 5 CFR updates
            logger.info("📋 Checking 5 CFR updates...")
            results['cfr_updates'] = await self.check_5cfr_updates()

            # Check training compliance
            logger.info("📚 Checking training compliance...")
            results['training_compliance'] = await self.check_training_compliance()

            # Generate action items
            if results['opm_updates']:
                results['actions_needed'].append(f"Review {len(results['opm_updates'])} new OPM updates")

            if results['cfr_updates']:
                results['actions_needed'].append("Review 5 CFR changes for compliance")

            compliance = results['training_compliance']
            if compliance.get('outdated_modules', 0) > 0:
                results['actions_needed'].append(f"Update {compliance['outdated_modules']} outdated training modules")

            if compliance.get('missing_modules'):
                results['actions_needed'].append(f"Add {len(compliance['missing_modules'])} missing training categories")

            # Save results
            results_file = Path("/opt/tower-echo-brain/logs/opm_monitoring.log")
            results_file.parent.mkdir(parents=True, exist_ok=True)

            with open(results_file, 'a') as f:
                f.write(json.dumps(results) + '\n')

            logger.info(f"✅ OPM/HR monitoring complete: {len(results['actions_needed'])} actions needed")

        except Exception as e:
            logger.error(f"Monitoring cycle failed: {e}")
            results['error'] = str(e)

        return results

    async def generate_compliance_report(self) -> str:
        """Generate HR compliance report"""
        results = await self.run_monitoring_cycle()

        report = f"""Federal HR Compliance Report
============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPM Updates:
-----------
"""
        for update in results['opm_updates'][:10]:
            report += f"• {update['title']} ({update['category']})\n"

        if not results['opm_updates']:
            report += "• No new updates\n"

        report += f"""
5 CFR Status:
------------
"""
        for update in results['cfr_updates']:
            report += f"• {update['title']}\n"

        if not results['cfr_updates']:
            report += "• No updates detected\n"

        compliance = results['training_compliance']
        report += f"""
Training Compliance:
-------------------
• Total Modules: {compliance.get('total_modules', 0)}
• Active: {compliance.get('active_modules', 0)}
• Outdated: {compliance.get('outdated_modules', 0)}
• Missing Categories: {', '.join(compliance.get('missing_modules', [])) or 'None'}

Actions Required:
----------------
"""
        for action in results['actions_needed']:
            report += f"⚠️ {action}\n"

        if not results['actions_needed']:
            report += "✅ No actions required - system compliant\n"

        return report

# Global instance
opm_monitor = OPMHRMonitor()