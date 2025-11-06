#!/usr/bin/env python3
"""
Comprehensive test suite for Character Evolution System
Tests database schema, API endpoints, and integration functionality
"""

import os
import json
import requests
import psycopg2
import psycopg2.extras
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterEvolutionTester:
    def __init__(self):
        self.base_url = "http://localhost:8350"
        self.db_config = {
            "host": "localhost",
            "database": "echo_brain",
            "user": os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick")),
            "password": "admin123"
        }
        self.test_results = []
        
    def test_result(self, test_name: str, passed: bool, details: str = ""):
        """Record test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = f"{status} - {test_name}: {details}"
        self.test_results.append((test_name, passed, details))
        print(result)
        
    def test_database_schema(self):
        """Test that all required tables exist with correct schema"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Test tables exist
            required_tables = [
                'character_evolution_timeline',
                'character_state_history', 
                'character_relationship_dynamics',
                'emotional_impact_tracking'
            ]
            
            for table in required_tables:
                cur.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table}'")
                if cur.fetchone()[0] == 1:
                    self.test_result(f"Database table {table}", True, "Table exists")
                else:
                    self.test_result(f"Database table {table}", False, "Table missing")
                    
            # Test indexes exist
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename IN ('character_evolution_timeline', 'character_state_history', 
                                   'character_relationship_dynamics', 'emotional_impact_tracking')
            """)
            indexes = cur.fetchall()
            self.test_result("Database indexes", len(indexes) >= 8, f"Found {len(indexes)} indexes")
            
        except Exception as e:
            self.test_result("Database schema", False, f"Error: {e}")
        finally:
            if conn:
                conn.close()
                
    def test_service_health(self):
        """Test service health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.test_result("Service health", True, f"Status: {data.get('status')}")
            else:
                self.test_result("Service health", False, f"Status code: {response.status_code}")
        except Exception as e:
            self.test_result("Service health", False, f"Error: {e}")
            
    def test_create_test_character(self):
        """Create a test character for evolution testing"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Insert test character
            cur.execute("""
                INSERT INTO anime_characters (name, description, personality, appearance, project_id)
                VALUES ('TestCharacter', 'A test character for evolution system', 
                       'Brave and determined', 'Young hero with blue hair', 'test_project')
                ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
                RETURNING id
            """)
            
            character_id = cur.fetchone()[0]
            conn.commit()
            
            self.test_character_id = character_id
            self.test_result("Create test character", True, f"Character ID: {character_id}")
            
        except Exception as e:
            self.test_result("Create test character", False, f"Error: {e}")
        finally:
            if conn:
                conn.close()
                
    def test_create_evolution_event(self):
        """Test creating character evolution event via API"""
        try:
            evolution_data = {
                "character_id": self.test_character_id,
                "evolution_type": "skill_gain",
                "title": "Learns Fire Magic",
                "description": "TestCharacter discovers their ability to control fire after intense training",
                "impact_level": 7,
                "previous_state": {"magic_level": 0, "skills": []},
                "new_state": {"magic_level": 3, "skills": ["fire_magic"]},
                "triggers": ["magical_training", "life_threatening_situation"]
            }
            
            response = requests.post(f"{self.base_url}/api/characters/evolution/", 
                                   json=evolution_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.evolution_id = data.get('evolution_id')
                    self.test_result("Create evolution event", True, f"Evolution ID: {self.evolution_id}")
                else:
                    self.test_result("Create evolution event", False, "API returned success=false")
            else:
                self.test_result("Create evolution event", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.test_result("Create evolution event", False, f"Error: {e}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Starting Character Evolution System Test Suite")
        print("=" * 60)
        
        self.test_database_schema()
        self.test_service_health()
        self.test_create_test_character()
        
        if hasattr(self, 'test_character_id'):
            self.test_create_evolution_event()
        
        # Summary
        print("\n" + "=" * 60)
        passed = sum(1 for _, passed, _ in self.test_results if passed)
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"📋 TEST SUMMARY:")
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("✅ CHARACTER EVOLUTION SYSTEM FULLY FUNCTIONAL!")
        elif success_rate >= 70:
            print("⚠️ CHARACTER EVOLUTION SYSTEM MOSTLY WORKING - Minor issues detected")
        else:
            print("❌ CHARACTER EVOLUTION SYSTEM HAS SIGNIFICANT ISSUES")
        
        return success_rate >= 90

if __name__ == "__main__":
    tester = CharacterEvolutionTester()
    tester.run_all_tests()