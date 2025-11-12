"""
Multi-Language Linting and Code Quality Module for Echo Brain
Handles Python, JavaScript, TypeScript, HTML, CSS, SQL, Go, Rust, etc.
"""

import os
import json
import subprocess
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MultiLanguageLinter:
    """Comprehensive linting for all Tower codebase languages"""

    def __init__(self):
        self.linters = {
            'python': {
                'extensions': ['.py'],
                'tools': ['pylint', 'black', 'ruff', 'mypy', 'autopep8', 'isort'],
                'command': 'pylint {file} --output-format=json --exit-zero'
            },
            'javascript': {
                'extensions': ['.js', '.jsx'],
                'tools': ['eslint', 'prettier', 'jshint'],
                'command': 'npx eslint {file} --format json || true'
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'tools': ['tslint', 'eslint', 'prettier'],
                'command': 'npx eslint {file} --format json || true'
            },
            'html': {
                'extensions': ['.html', '.htm'],
                'tools': ['htmlhint', 'tidy'],
                'command': 'npx htmlhint {file} --format json || true'
            },
            'css': {
                'extensions': ['.css', '.scss', '.sass', '.less'],
                'tools': ['stylelint', 'csslint'],
                'command': 'npx stylelint {file} --formatter json || true'
            },
            'sql': {
                'extensions': ['.sql'],
                'tools': ['sqlfluff'],
                'command': 'sqlfluff lint {file} --format json || true'
            },
            'go': {
                'extensions': ['.go'],
                'tools': ['golint', 'go vet'],
                'command': 'golint {file} 2>&1 || true'
            },
            'rust': {
                'extensions': ['.rs'],
                'tools': ['cargo clippy'],
                'command': 'cargo clippy --message-format json 2>&1 || true'
            },
            'docker': {
                'extensions': ['Dockerfile', '.dockerfile'],
                'tools': ['hadolint'],
                'command': 'hadolint {file} --format json || true'
            },
            'yaml': {
                'extensions': ['.yml', '.yaml'],
                'tools': ['yamllint'],
                'command': 'yamllint {file} --format json || true'
            },
            'json': {
                'extensions': ['.json'],
                'tools': ['jsonlint'],
                'command': 'python3 -m json.tool {file} > /dev/null 2>&1 && echo "valid" || echo "invalid"'
            }
        }

        self.fix_commands = {
            'python': 'black {file} && isort {file} && autopep8 --in-place {file}',
            'javascript': 'npx prettier --write {file} && npx eslint --fix {file}',
            'typescript': 'npx prettier --write {file} && npx eslint --fix {file}',
            'html': 'npx prettier --write {file}',
            'css': 'npx prettier --write {file} && npx stylelint --fix {file}',
            'json': 'python3 -c "import json; f=open(\'{file}\'); d=json.load(f); f.close(); f=open(\'{file}\', \'w\'); json.dump(d, f, indent=2); f.close()"',
            'yaml': 'yamllint --fix {file}'
        }

    async def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single file for code quality issues"""
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            if os.path.basename(filepath) in ['Dockerfile']:
                language = 'docker'
            else:
                return {'file': filepath, 'language': 'unknown', 'issues': [], 'score': 10.0}

        language = None
        for lang, config in self.linters.items():
            if ext in config['extensions']:
                language = lang
                break

        if not language:
            return {'file': filepath, 'language': 'unknown', 'issues': [], 'score': 10.0}

        command = self.linters[language]['command'].format(file=filepath)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            issues = self._parse_linter_output(result.stdout, language)
            score = self._calculate_score(issues)

            return {
                'file': filepath,
                'language': language,
                'issues': issues,
                'score': score,
                'fixable': language in self.fix_commands
            }
        except Exception as e:
            logger.error(f"Error analyzing {filepath}: {e}")
            return {'file': filepath, 'language': language, 'error': str(e), 'score': 0}

    async def fix_file(self, filepath: str) -> bool:
        """Automatically fix issues in a file"""
        ext = os.path.splitext(filepath)[1].lower()
        language = None

        for lang, config in self.linters.items():
            if ext in config['extensions']:
                language = lang
                break

        if language not in self.fix_commands:
            return False

        command = self.fix_commands[language].format(file=filepath)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error fixing {filepath}: {e}")
            return False

    def _parse_linter_output(self, output: str, language: str) -> List[Dict]:
        """Parse linter output into structured issues"""
        issues = []

        if language == 'python':
            try:
                data = json.loads(output) if output else []
                for issue in data:
                    issues.append({
                        'type': issue.get('type', 'warning'),
                        'line': issue.get('line', 0),
                        'column': issue.get('column', 0),
                        'message': issue.get('message', ''),
                        'rule': issue.get('symbol', '')
                    })
            except:
                pass

        elif language in ['javascript', 'typescript']:
            try:
                data = json.loads(output) if output else []
                if isinstance(data, list) and data:
                    for file_result in data:
                        for msg in file_result.get('messages', []):
                            issues.append({
                                'type': 'error' if msg.get('severity') == 2 else 'warning',
                                'line': msg.get('line', 0),
                                'column': msg.get('column', 0),
                                'message': msg.get('message', ''),
                                'rule': msg.get('ruleId', '')
                            })
            except:
                pass

        return issues

    def _calculate_score(self, issues: List[Dict]) -> float:
        """Calculate quality score based on issues (0-10 scale)"""
        if not issues:
            return 10.0

        score = 10.0
        for issue in issues:
            if issue.get('type') == 'error':
                score -= 0.5
            elif issue.get('type') == 'warning':
                score -= 0.1
            else:
                score -= 0.05

        return max(0, score)

    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze entire project for code quality"""
        results = {
            'project': project_path,
            'timestamp': datetime.now().isoformat(),
            'languages': {},
            'total_files': 0,
            'total_issues': 0,
            'average_score': 0,
            'fixable_files': []
        }

        all_scores = []

        for root, _, files in os.walk(project_path):
            for file in files:
                filepath = os.path.join(root, file)

                # Skip common non-code files
                if any(skip in filepath for skip in ['node_modules', '.git', '__pycache__', '.pyc']):
                    continue

                result = await self.analyze_file(filepath)

                if result.get('language') != 'unknown':
                    language = result['language']

                    if language not in results['languages']:
                        results['languages'][language] = {
                            'files': 0,
                            'issues': 0,
                            'scores': []
                        }

                    results['languages'][language]['files'] += 1
                    results['languages'][language]['issues'] += len(result.get('issues', []))
                    results['languages'][language]['scores'].append(result.get('score', 0))

                    results['total_files'] += 1
                    results['total_issues'] += len(result.get('issues', []))
                    all_scores.append(result.get('score', 0))

                    if result.get('fixable') and result.get('score', 10) < 7:
                        results['fixable_files'].append(filepath)

        if all_scores:
            results['average_score'] = sum(all_scores) / len(all_scores)

        # Calculate average scores per language
        for lang_data in results['languages'].values():
            if lang_data['scores']:
                lang_data['average_score'] = sum(lang_data['scores']) / len(lang_data['scores'])

        return results

    async def create_refactor_task(self, filepath: str, issues: List[Dict]) -> Dict:
        """Create a refactoring task for Echo's task queue"""
        return {
            'task_type': 'CODE_REFACTOR',
            'priority': 'HIGH' if any(i.get('type') == 'error' for i in issues) else 'NORMAL',
            'payload': {
                'file': filepath,
                'issues': issues,
                'auto_fix': True,
                'language': self._detect_language(filepath)
            }
        }

    def _detect_language(self, filepath: str) -> str:
        """Detect language from file extension"""
        ext = os.path.splitext(filepath)[1].lower()
        for lang, config in self.linters.items():
            if ext in config['extensions']:
                return lang
        return 'unknown'

    async def install_missing_tools(self) -> Dict[str, bool]:
        """Install missing linting tools"""
        installed = {}

        # Python tools
        python_tools = ['pylint', 'black', 'ruff', 'mypy', 'autopep8', 'isort', 'sqlfluff']
        for tool in python_tools:
            try:
                subprocess.run(f'pip install {tool}', shell=True, check=True, capture_output=True)
                installed[tool] = True
            except:
                installed[tool] = False

        # Node.js tools
        npm_tools = ['eslint', 'prettier', 'stylelint', 'htmlhint', 'jsonlint', 'tslint']
        for tool in npm_tools:
            try:
                subprocess.run(f'npm install -g {tool}', shell=True, check=True, capture_output=True)
                installed[tool] = True
            except:
                installed[tool] = False

        # System tools
        system_tools = [
            ('golint', 'go install golang.org/x/lint/golint@latest'),
            ('hadolint', 'wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64'),
            ('yamllint', 'pip install yamllint')
        ]

        for tool, install_cmd in system_tools:
            try:
                subprocess.run(install_cmd, shell=True, check=True, capture_output=True)
                installed[tool] = True
            except:
                installed[tool] = False

        return installed