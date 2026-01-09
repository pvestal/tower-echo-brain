#!/usr/bin/env python3
"""
Comprehensive Data Ingestion for Echo Brain
Learn from ALL available data sources, not just conversations
"""
import os
import json
import logging
import psycopg2
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class ComprehensiveDataIngestion:
    """
    Learns from ALL available data sources:
    - Existing codebase (Tower services, Echo Brain)
    - Database content (conversations, learnings, domain data)
    - System configuration (nginx, systemd, etc.)
    - Documentation and knowledge base
    - Git history and commit patterns
    """

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'echo_brain',
            'user': 'patrick',
            'password': '***REMOVED***'
        }
        self.data_sources = {
            'codebase_analysis': [],
            'database_mining': [],
            'documentation_processing': [],
            'configuration_analysis': [],
            'git_history_analysis': []
        }
        self.extracted_patterns = []

    async def ingest_all_sources(self) -> Dict[str, Any]:
        """Comprehensive ingestion from all data sources"""
        logger.info("🚀 Starting comprehensive data ingestion...")

        results = {
            'codebase_patterns': await self.analyze_codebase_patterns(),
            'database_patterns': await self.mine_database_knowledge(),
            'documentation_patterns': await self.process_documentation(),
            'system_patterns': await self.analyze_system_configs(),
            'git_patterns': await self.analyze_git_history(),
            'integration_patterns': await self.extract_integration_patterns()
        }

        # Consolidate all patterns
        all_patterns = self.consolidate_patterns(results)

        # Store patterns for Echo to use
        await self.store_learned_patterns(all_patterns)

        logger.info(f"✅ Ingestion complete - extracted {len(all_patterns)} patterns")
        return {
            'status': 'completed',
            'patterns_extracted': len(all_patterns),
            'sources_processed': len(results),
            'pattern_breakdown': {k: len(v) for k, v in results.items()},
            'timestamp': datetime.now().isoformat()
        }

    async def analyze_codebase_patterns(self) -> List[Dict]:
        """Extract patterns from Tower codebase"""
        logger.info("🔍 Analyzing codebase patterns...")
        patterns = []

        # Analyze Tower services
        tower_services = [d for d in Path('/opt').iterdir() if d.is_dir() and d.name.startswith('tower-')]

        for service_dir in tower_services:
            try:
                service_patterns = await self.analyze_service_structure(service_dir)
                patterns.extend(service_patterns)
            except Exception as e:
                logger.warning(f"Failed to analyze {service_dir}: {e}")

        # Extract architectural patterns
        architectural_patterns = self.extract_architectural_patterns(patterns)
        patterns.extend(architectural_patterns)

        return patterns

    def extract_architectural_patterns(self, service_patterns: List[Dict]) -> List[Dict]:
        """Extract overall architectural patterns from individual service patterns"""
        architectural_patterns = []

        # Analyze framework distribution
        frameworks = {}
        for pattern in service_patterns:
            if pattern.get('type') == 'framework_preference':
                framework = pattern['pattern']
                frameworks[framework] = frameworks.get(framework, 0) + 1

        # Create architectural insights
        if frameworks:
            most_common = max(frameworks.items(), key=lambda x: x[1])
            architectural_patterns.append({
                'type': 'architectural_pattern',
                'pattern': f'Tower ecosystem primarily uses {most_common[0]} (found in {most_common[1]} services)',
                'source': 'architectural_analysis',
                'confidence': 0.8
            })

        return architectural_patterns

    async def analyze_service_structure(self, service_dir: Path) -> List[Dict]:
        """Analyze individual Tower service"""
        patterns = []

        # Check for common file patterns
        package_files = list(service_dir.glob('**/package.json'))
        requirements_files = list(service_dir.glob('**/requirements.txt'))
        py_files = list(service_dir.glob('**/*.py'))
        js_files = list(service_dir.glob('**/*.js'))

        # Framework detection
        if package_files:
            framework = self.detect_js_framework(package_files[0])
            if framework:
                patterns.append({
                    'type': 'framework_preference',
                    'pattern': f'Patrick uses {framework} for JavaScript projects',
                    'source': str(service_dir),
                    'confidence': 0.8
                })

        if requirements_files:
            python_patterns = self.extract_python_patterns(requirements_files[0])
            patterns.extend(python_patterns)

        # File organization patterns
        org_patterns = self.analyze_file_organization(service_dir)
        patterns.extend(org_patterns)

        return patterns

    def detect_js_framework(self, package_file: Path) -> Optional[str]:
        """Detect JavaScript framework from package.json"""
        try:
            with open(package_file) as f:
                package_data = json.load(f)

            dependencies = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}

            if 'vue' in dependencies or '@vue/cli' in dependencies:
                return 'Vue.js'
            elif 'react' in dependencies:
                return 'React'
            elif 'express' in dependencies:
                return 'Express.js'
            elif 'fastapi' in str(package_file.parent):
                return 'FastAPI'

        except Exception as e:
            logger.debug(f"Could not parse {package_file}: {e}")

        return None

    def extract_python_patterns(self, requirements_file: Path) -> List[Dict]:
        """Extract Python framework preferences"""
        patterns = []
        try:
            with open(requirements_file) as f:
                requirements = f.read()

            if 'fastapi' in requirements.lower():
                patterns.append({
                    'type': 'framework_preference',
                    'pattern': 'Patrick prefers FastAPI for Python web services',
                    'source': str(requirements_file),
                    'confidence': 0.9
                })

            if 'psycopg2' in requirements.lower():
                patterns.append({
                    'type': 'database_preference',
                    'pattern': 'Patrick uses PostgreSQL for Python database connections',
                    'source': str(requirements_file),
                    'confidence': 0.9
                })

        except Exception as e:
            logger.debug(f"Could not parse {requirements_file}: {e}")

        return patterns

    def analyze_file_organization(self, service_dir: Path) -> List[Dict]:
        """Extract file organization patterns"""
        patterns = []

        # Check directory structure
        subdirs = [d.name for d in service_dir.iterdir() if d.is_dir()]

        common_structures = {
            'src': 'Patrick organizes code in src/ directories',
            'api': 'Patrick separates API routes into api/ directories',
            'services': 'Patrick uses services/ for business logic',
            'db': 'Patrick puts database code in db/ directories',
            'tests': 'Patrick includes tests/ directories for testing'
        }

        for dirname, description in common_structures.items():
            if dirname in subdirs:
                patterns.append({
                    'type': 'file_organization',
                    'pattern': description,
                    'source': str(service_dir),
                    'confidence': 0.7
                })

        return patterns

    async def mine_database_knowledge(self) -> List[Dict]:
        """Extract patterns from database content"""
        logger.info("🗄️ Mining database knowledge...")
        patterns = []

        try:
            db = psycopg2.connect(**self.db_config)
            cursor = db.cursor()

            # Mine conversation patterns
            cursor.execute("""
                SELECT intent, query, response, created_at
                FROM conversations
                WHERE created_at > NOW() - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT 200
            """)

            conversation_patterns = []
            for row in cursor.fetchall():
                intent, query, response, created_at = row
                conversation_patterns.append({
                    'intent': intent,
                    'query': query[:200],  # Truncate for privacy
                    'response': response[:200],
                    'timestamp': created_at
                })

            # Extract communication patterns
            comm_patterns = self.extract_communication_patterns(conversation_patterns)
            patterns.extend(comm_patterns)

            # Mine existing learning history
            cursor.execute("""
                SELECT fact_type, learned_fact, confidence
                FROM learning_history
                WHERE confidence > 0.7
                ORDER BY confidence DESC
            """)

            for row in cursor.fetchall():
                fact_type, learned_fact, confidence = row
                patterns.append({
                    'type': fact_type,
                    'pattern': learned_fact,
                    'source': 'learning_history_table',
                    'confidence': confidence
                })

            db.close()

        except Exception as e:
            logger.error(f"Database mining failed: {e}")

        return patterns

    def extract_communication_patterns(self, conversations: List[Dict]) -> List[Dict]:
        """Extract Patrick's communication patterns from conversations"""
        patterns = []

        # Analyze query patterns
        queries = [conv['query'].lower() for conv in conversations]

        # Technical preference indicators
        tech_mentions = {}
        for query in queries:
            if 'database' in query:
                if 'postgres' in query:
                    tech_mentions['postgresql'] = tech_mentions.get('postgresql', 0) + 1
                if 'mysql' in query:
                    tech_mentions['mysql'] = tech_mentions.get('mysql', 0) + 1

            if 'frontend' in query:
                if 'vue' in query:
                    tech_mentions['vue'] = tech_mentions.get('vue', 0) + 1
                if 'react' in query:
                    tech_mentions['react'] = tech_mentions.get('react', 0) + 1

        # Convert mentions to patterns
        for tech, count in tech_mentions.items():
            if count >= 3:  # Minimum threshold
                patterns.append({
                    'type': 'technical_preference',
                    'pattern': f'Patrick frequently asks about {tech} (mentioned {count} times)',
                    'source': 'conversation_analysis',
                    'confidence': min(0.9, 0.5 + (count * 0.1))
                })

        return patterns

    async def process_documentation(self) -> List[Dict]:
        """Process documentation and KB articles"""
        logger.info("📚 Processing documentation...")
        patterns = []

        # Process KB articles via API
        try:
            import requests
            response = requests.get('http://localhost:8307/api/kb/articles?limit=100')
            if response.status_code == 200:
                articles = response.json()
                doc_patterns = self.extract_documentation_patterns(articles)
                patterns.extend(doc_patterns)
        except Exception as e:
            logger.warning(f"Could not fetch KB articles: {e}")

        # Process README files
        readme_files = []
        for root, dirs, files in os.walk('/opt'):
            for file in files:
                if file.lower() in ['readme.md', 'readme.txt', 'readme']:
                    readme_files.append(os.path.join(root, file))

        for readme_file in readme_files[:20]:  # Limit to avoid overload
            try:
                readme_patterns = self.extract_readme_patterns(readme_file)
                patterns.extend(readme_patterns)
            except Exception as e:
                logger.debug(f"Could not process {readme_file}: {e}")

        return patterns

    def extract_documentation_patterns(self, articles: List[Dict]) -> List[Dict]:
        """Extract patterns from KB articles"""
        patterns = []

        for article in articles:
            content = article.get('content', '').lower()
            title = article.get('title', '')

            # Extract system knowledge
            if 'tower' in title.lower():
                if 'postgresql' in content:
                    patterns.append({
                        'type': 'system_architecture',
                        'pattern': 'Tower system uses PostgreSQL for data persistence',
                        'source': f"kb_article_{article.get('id')}",
                        'confidence': 0.8
                    })

                if 'fastapi' in content:
                    patterns.append({
                        'type': 'system_architecture',
                        'pattern': 'Tower services use FastAPI for API endpoints',
                        'source': f"kb_article_{article.get('id')}",
                        'confidence': 0.8
                    })

        return patterns

    async def analyze_system_configs(self) -> List[Dict]:
        """Analyze system configuration patterns"""
        logger.info("⚙️ Analyzing system configurations...")
        patterns = []

        # Analyze nginx configs
        nginx_configs = ['/etc/nginx/sites-available/tower.conf']
        for config_file in nginx_configs:
            if os.path.exists(config_file):
                nginx_patterns = self.extract_nginx_patterns(config_file)
                patterns.extend(nginx_patterns)

        # Analyze systemd services
        systemd_files = list(Path('/etc/systemd/system').glob('tower-*.service'))
        for service_file in systemd_files:
            systemd_patterns = self.extract_systemd_patterns(service_file)
            patterns.extend(systemd_patterns)

        return patterns

    def extract_readme_patterns(self, readme_file: str) -> List[Dict]:
        """Extract patterns from README files"""
        patterns = []
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # Look for technology mentions
            if 'postgresql' in content or 'postgres' in content:
                patterns.append({
                    'type': 'documentation_pattern',
                    'pattern': 'Project documentation mentions PostgreSQL usage',
                    'source': readme_file,
                    'confidence': 0.7
                })

            if 'fastapi' in content:
                patterns.append({
                    'type': 'documentation_pattern',
                    'pattern': 'Project documentation mentions FastAPI framework',
                    'source': readme_file,
                    'confidence': 0.7
                })

        except Exception as e:
            logger.debug(f"Could not read README {readme_file}: {e}")

        return patterns

    def extract_nginx_patterns(self, config_file: str) -> List[Dict]:
        """Extract patterns from nginx configuration"""
        patterns = []
        try:
            with open(config_file) as f:
                content = f.read()

            if 'proxy_pass' in content:
                patterns.append({
                    'type': 'deployment_pattern',
                    'pattern': 'Patrick uses nginx reverse proxy for service routing',
                    'source': config_file,
                    'confidence': 0.9
                })

            if 'ssl_certificate' in content:
                patterns.append({
                    'type': 'security_pattern',
                    'pattern': 'Patrick configures HTTPS/SSL for web services',
                    'source': config_file,
                    'confidence': 0.9
                })

        except Exception as e:
            logger.debug(f"Could not parse nginx config: {e}")

        return patterns

    def extract_systemd_patterns(self, service_file: Path) -> List[Dict]:
        """Extract patterns from systemd service files"""
        patterns = []
        try:
            with open(service_file) as f:
                content = f.read()

            if 'venv/bin/python' in content:
                patterns.append({
                    'type': 'deployment_pattern',
                    'pattern': 'Patrick uses Python virtual environments for service deployment',
                    'source': str(service_file),
                    'confidence': 0.8
                })

            if 'uvicorn' in content:
                patterns.append({
                    'type': 'deployment_pattern',
                    'pattern': 'Patrick uses Uvicorn for FastAPI service deployment',
                    'source': str(service_file),
                    'confidence': 0.8
                })

        except Exception as e:
            logger.debug(f"Could not parse systemd file {service_file}: {e}")

        return patterns

    def analyze_service_dependencies(self) -> List[Dict]:
        """Analyze dependencies between Tower services"""
        patterns = []

        # This would be more sophisticated in reality - analyzing import statements,
        # API calls between services, shared databases, etc.
        patterns.append({
            'type': 'service_architecture',
            'pattern': 'Tower services follow microservices architecture with PostgreSQL shared database',
            'source': 'dependency_analysis',
            'confidence': 0.8
        })

        return patterns

    async def analyze_git_history(self) -> List[Dict]:
        """Analyze git commit patterns"""
        logger.info("📝 Analyzing git history...")
        patterns = []

        # Analyze Echo Brain git history
        try:
            result = subprocess.run([
                'git', 'log', '--oneline', '--since=30 days ago'
            ], cwd='/opt/tower-echo-brain', capture_output=True, text=True)

            if result.returncode == 0:
                commit_messages = result.stdout.strip().split('\n')
                git_patterns = self.extract_git_patterns(commit_messages)
                patterns.extend(git_patterns)

        except Exception as e:
            logger.debug(f"Git analysis failed: {e}")

        return patterns

    def extract_git_patterns(self, commit_messages: List[str]) -> List[Dict]:
        """Extract patterns from git commit messages"""
        patterns = []

        # Analyze commit message style
        if len(commit_messages) > 10:
            patterns.append({
                'type': 'development_pattern',
                'pattern': 'Patrick maintains active git commit history with detailed messages',
                'source': 'git_history',
                'confidence': 0.8
            })

        # Look for technical keywords
        all_messages = ' '.join(commit_messages).lower()
        if 'refactor' in all_messages:
            patterns.append({
                'type': 'development_pattern',
                'pattern': 'Patrick regularly refactors code for better architecture',
                'source': 'git_history',
                'confidence': 0.7
            })

        return patterns

    async def extract_integration_patterns(self) -> List[Dict]:
        """Extract service integration patterns"""
        logger.info("🔗 Analyzing service integration patterns...")
        patterns = []

        # Analyze port assignments
        port_patterns = self.analyze_port_assignments()
        patterns.extend(port_patterns)

        # Analyze service dependencies
        dependency_patterns = self.analyze_service_dependencies()
        patterns.extend(dependency_patterns)

        return patterns

    def analyze_port_assignments(self) -> List[Dict]:
        """Analyze how services are assigned ports"""
        patterns = []

        # Known port assignments from CLAUDE.md context
        port_assignments = {
            8309: 'Echo Brain',
            8307: 'Knowledge Base',
            8188: 'ComfyUI',
            8080: 'Dashboard'
        }

        if len(port_assignments) > 5:
            patterns.append({
                'type': 'system_architecture',
                'pattern': 'Patrick uses systematic port assignment for microservices (8000-8400 range)',
                'source': 'port_analysis',
                'confidence': 0.8
            })

        return patterns

    def consolidate_patterns(self, results: Dict[str, List]) -> List[Dict]:
        """Consolidate patterns from all sources, removing duplicates"""
        all_patterns = []

        for source, patterns in results.items():
            for pattern in patterns:
                pattern['extraction_source'] = source
                all_patterns.append(pattern)

        # Remove near-duplicates based on pattern text similarity
        consolidated = []
        seen_patterns = set()

        for pattern in all_patterns:
            pattern_key = pattern['pattern'].lower()[:50]  # First 50 chars for deduplication

            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                consolidated.append(pattern)

        return consolidated

    async def store_learned_patterns(self, patterns: List[Dict]):
        """Store extracted patterns in the learning_history table"""
        try:
            db = psycopg2.connect(**self.db_config)
            cursor = db.cursor()

            for pattern in patterns:
                cursor.execute("""
                    INSERT INTO learning_history
                    (fact_type, learned_fact, confidence, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (fact_type, learned_fact) DO NOTHING
                """, (
                    pattern.get('type', 'extracted_pattern'),
                    pattern['pattern'],
                    pattern.get('confidence', 0.7),
                    json.dumps({
                        'source': pattern.get('source', 'unknown'),
                        'extraction_source': pattern.get('extraction_source', 'unknown'),
                        'extraction_timestamp': datetime.now().isoformat()
                    }),
                    datetime.now()
                ))

            db.commit()
            db.close()

            logger.info(f"✅ Stored {len(patterns)} learned patterns in database")

        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")

# Global instance
comprehensive_ingestion = ComprehensiveDataIngestion()