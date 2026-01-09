#!/usr/bin/env python3
"""
Quality Reporting System for Echo Video Generation
Creates detailed reports and tracks quality improvements over time
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityReporter:
    def __init__(self, db_path: str = None):
        """Initialize the quality reporting system"""
        if db_path is None:
            db_path = "/home/patrick/Documents/Tower/core-services/echo-brain/quality_scores.json"

        self.db_path = db_path
        self.load_database()

    def load_database(self):
        """Load quality database"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    self.db = json.load(f)
            else:
                self.db = {
                    "assessments": [],
                    "statistics": {},
                    "trends": [],
                    "settings": {
                        "quality_threshold": 70,
                        "target_score": 90
                    }
                }
            logger.info(f"📊 Quality database loaded: {len(self.db.get('assessments', []))} records")
        except Exception as e:
            logger.error(f"❌ Failed to load quality database: {e}")
            self.db = {"assessments": [], "statistics": {}, "trends": []}

    def save_database(self):
        """Save quality database"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.db, f, indent=2)
            logger.info("💾 Quality database saved")
        except Exception as e:
            logger.error(f"❌ Failed to save quality database: {e}")

    def add_assessment(self, assessment: Dict[str, Any]):
        """Add a new quality assessment to the database"""
        # Ensure required fields
        assessment.setdefault("timestamp", datetime.now().isoformat())
        assessment.setdefault("overall_score", 0)

        # Add to assessments
        self.db["assessments"].append(assessment)

        # Update statistics
        self.update_statistics()

        # Save database
        self.save_database()

        logger.info(f"✅ Assessment added: Score {assessment.get('overall_score', 0):.1f}")

    def update_statistics(self):
        """Update overall statistics"""
        assessments = self.db.get("assessments", [])

        if not assessments:
            return

        scores = [a.get("overall_score", 0) for a in assessments]
        pass_count = sum(1 for a in assessments if a.get("pass_threshold", False))

        self.db["statistics"] = {
            "total_assessments": len(assessments),
            "average_score": sum(scores) / len(scores),
            "highest_score": max(scores),
            "lowest_score": min(scores),
            "pass_rate": pass_count / len(assessments) * 100,
            "last_updated": datetime.now().isoformat()
        }

        # Calculate trends (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_assessments = [
            a for a in assessments
            if datetime.fromisoformat(a.get("timestamp", "")) > recent_cutoff
        ]

        if recent_assessments:
            recent_scores = [a.get("overall_score", 0) for a in recent_assessments]
            self.db["statistics"]["recent_average"] = sum(recent_scores) / len(recent_scores)
            self.db["statistics"]["recent_count"] = len(recent_assessments)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive quality report"""
        assessments = self.db.get("assessments", [])
        stats = self.db.get("statistics", {})

        if not assessments:
            return {
                "error": "No assessments found",
                "message": "Generate some videos first to see quality reports"
            }

        # Overall statistics
        report = {
            "generation_time": datetime.now().isoformat(),
            "summary": {
                "total_videos": stats.get("total_assessments", 0),
                "average_score": round(stats.get("average_score", 0), 2),
                "pass_rate": round(stats.get("pass_rate", 0), 2),
                "quality_grade": self._get_quality_grade(stats.get("average_score", 0))
            }
        }

        # Score distribution
        scores = [a.get("overall_score", 0) for a in assessments]
        report["score_distribution"] = {
            "excellent": len([s for s in scores if s >= 90]),
            "good": len([s for s in scores if 70 <= s < 90]),
            "fair": len([s for s in scores if 50 <= s < 70]),
            "poor": len([s for s in scores if s < 50])
        }

        # Technical quality breakdown
        tech_breakdown = self._analyze_technical_quality(assessments)
        report["technical_analysis"] = tech_breakdown

        # Trends and improvements
        trends = self._analyze_trends(assessments)
        report["trends"] = trends

        # Top performing videos
        top_videos = sorted(assessments, key=lambda x: x.get("overall_score", 0), reverse=True)[:5]
        report["top_videos"] = [
            {
                "video_path": v.get("video_path", ""),
                "score": v.get("overall_score", 0),
                "prompt": v.get("prompt", "")[:100] + "..." if len(v.get("prompt", "")) > 100 else v.get("prompt", ""),
                "timestamp": v.get("timestamp", "")
            }
            for v in top_videos
        ]

        # Recent improvements
        recent_assessments = sorted(assessments, key=lambda x: x.get("timestamp", ""))[-10:]
        if len(recent_assessments) >= 2:
            recent_avg = sum(a.get("overall_score", 0) for a in recent_assessments[-5:]) / 5
            older_avg = sum(a.get("overall_score", 0) for a in recent_assessments[-10:-5]) / 5
            improvement = recent_avg - older_avg

            report["recent_improvement"] = {
                "score_change": round(improvement, 2),
                "improving": improvement > 0,
                "recent_average": round(recent_avg, 2),
                "previous_average": round(older_avg, 2)
            }

        # Recommendations
        report["recommendations"] = self._generate_system_recommendations(assessments, stats)

        return report

    def _get_quality_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _analyze_technical_quality(self, assessments: List[Dict]) -> Dict[str, Any]:
        """Analyze technical quality components across all assessments"""
        tech_components = ["resolution", "frame_consistency", "sharpness", "noise", "artifacts", "encoding_quality", "bitrate"]
        analysis = {}

        for component in tech_components:
            scores = []
            for assessment in assessments:
                tech_quality = assessment.get("technical_quality", {})
                if component in tech_quality:
                    scores.append(tech_quality[component].get("score", 0))

            if scores:
                analysis[component] = {
                    "average": round(sum(scores) / len(scores), 2),
                    "best": max(scores),
                    "worst": min(scores),
                    "count": len(scores),
                    "needs_improvement": sum(scores) / len(scores) < 70
                }

        return analysis

    def _analyze_trends(self, assessments: List[Dict]) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if len(assessments) < 5:
            return {"message": "Need more assessments to analyze trends"}

        # Sort by timestamp
        sorted_assessments = sorted(assessments, key=lambda x: x.get("timestamp", ""))

        # Calculate moving averages
        window_size = min(5, len(sorted_assessments))
        moving_averages = []

        for i in range(len(sorted_assessments) - window_size + 1):
            window = sorted_assessments[i:i + window_size]
            avg_score = sum(a.get("overall_score", 0) for a in window) / window_size
            moving_averages.append({
                "timestamp": window[-1].get("timestamp", ""),
                "average_score": round(avg_score, 2)
            })

        # Determine trend direction
        if len(moving_averages) >= 2:
            recent_trend = moving_averages[-1]["average_score"] - moving_averages[0]["average_score"]
            trend_direction = "improving" if recent_trend > 0 else "declining" if recent_trend < 0 else "stable"
        else:
            trend_direction = "insufficient_data"

        return {
            "trend_direction": trend_direction,
            "trend_magnitude": round(recent_trend, 2) if len(moving_averages) >= 2 else 0,
            "moving_averages": moving_averages[-10:],  # Last 10 data points
            "stability": "stable" if abs(recent_trend) < 2 else "volatile"
        }

    def _generate_system_recommendations(self, assessments: List[Dict], stats: Dict) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []

        avg_score = stats.get("average_score", 0)
        pass_rate = stats.get("pass_rate", 0)

        # Overall performance recommendations
        if avg_score < 60:
            recommendations.append("🚨 CRITICAL: Average quality is very low. Review generation pipeline and parameters.")
        elif avg_score < 70:
            recommendations.append("⚠️ WARNING: Quality below acceptable threshold. Focus on technical improvements.")
        elif avg_score < 85:
            recommendations.append("📈 GOOD: Solid foundation. Fine-tune for excellence.")
        else:
            recommendations.append("🌟 EXCELLENT: Maintaining high quality standards!")

        # Pass rate recommendations
        if pass_rate < 50:
            recommendations.append("🎯 PRIORITY: Less than half of videos meet quality standards. Increase generation parameters.")
        elif pass_rate < 80:
            recommendations.append("🔧 IMPROVE: Aim for 80%+ pass rate. Consider adjusting quality thresholds.")

        # Technical component recommendations
        tech_analysis = self._analyze_technical_quality(assessments)
        weak_components = [comp for comp, data in tech_analysis.items() if data.get("needs_improvement", False)]

        if weak_components:
            recommendations.append(f"🔧 FOCUS AREAS: Improve {', '.join(weak_components)} quality components.")

        # Recent trend recommendations
        recent_assessments = sorted(assessments, key=lambda x: x.get("timestamp", ""))[-5:]
        if len(recent_assessments) >= 3:
            recent_scores = [a.get("overall_score", 0) for a in recent_assessments]
            if recent_scores[-1] < recent_scores[0]:
                recommendations.append("📉 ATTENTION: Recent quality decline detected. Check for system changes.")

        return recommendations

    def generate_video_specific_report(self, video_path: str) -> Dict[str, Any]:
        """Generate detailed report for a specific video"""
        assessments = self.db.get("assessments", [])

        # Find assessment for this video
        video_assessment = None
        for assessment in assessments:
            if assessment.get("video_path") == video_path:
                video_assessment = assessment
                break

        if not video_assessment:
            return {"error": f"No assessment found for video: {video_path}"}

        # Compare to database averages
        all_scores = [a.get("overall_score", 0) for a in assessments]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        report = {
            "video_path": video_path,
            "assessment": video_assessment,
            "comparison": {
                "score_vs_average": round(video_assessment.get("overall_score", 0) - avg_score, 2),
                "percentile": self._calculate_percentile(video_assessment.get("overall_score", 0), all_scores),
                "rank": sorted(all_scores, reverse=True).index(video_assessment.get("overall_score", 0)) + 1
            },
            "improvement_suggestions": video_assessment.get("recommendations", [])
        }

        return report

    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate what percentile this score represents"""
        if not all_scores:
            return 0

        sorted_scores = sorted(all_scores)
        rank = sum(1 for s in sorted_scores if s <= score)
        return round((rank / len(sorted_scores)) * 100, 1)

    def export_report_to_file(self, report: Dict[str, Any], filename: str = None) -> str:
        """Export report to a formatted file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/patrick/Documents/Tower/core-services/echo-brain/quality_report_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)

            # Also create a human-readable summary
            txt_filename = filename.replace('.json', '_summary.txt')
            self._create_text_summary(report, txt_filename)

            logger.info(f"📄 Report exported to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Failed to export report: {e}")
            return ""

    def _create_text_summary(self, report: Dict[str, Any], filename: str):
        """Create a human-readable text summary"""
        try:
            with open(filename, 'w') as f:
                f.write("ECHO VIDEO QUALITY REPORT\n")
                f.write("=" * 50 + "\n\n")

                summary = report.get("summary", {})
                f.write(f"📊 OVERALL STATISTICS\n")
                f.write(f"   Total Videos: {summary.get('total_videos', 0)}\n")
                f.write(f"   Average Score: {summary.get('average_score', 0)}/100\n")
                f.write(f"   Pass Rate: {summary.get('pass_rate', 0)}%\n")
                f.write(f"   Quality Grade: {summary.get('quality_grade', 'N/A')}\n\n")

                if "score_distribution" in report:
                    dist = report["score_distribution"]
                    f.write(f"📈 SCORE DISTRIBUTION\n")
                    f.write(f"   Excellent (90+): {dist.get('excellent', 0)} videos\n")
                    f.write(f"   Good (70-89): {dist.get('good', 0)} videos\n")
                    f.write(f"   Fair (50-69): {dist.get('fair', 0)} videos\n")
                    f.write(f"   Poor (<50): {dist.get('poor', 0)} videos\n\n")

                if "recommendations" in report:
                    f.write(f"💡 SYSTEM RECOMMENDATIONS\n")
                    for i, rec in enumerate(report["recommendations"], 1):
                        f.write(f"   {i}. {rec}\n")

                f.write(f"\nGenerated: {report.get('generation_time', 'Unknown')}\n")

        except Exception as e:
            logger.error(f"❌ Failed to create text summary: {e}")

# Example usage and testing
def test_quality_reporter():
    """Test the quality reporting system"""
    reporter = QualityReporter()

    # Add some sample assessments for testing
    sample_assessments = [
        {
            "video_path": "/test/video1.mp4",
            "overall_score": 85.2,
            "pass_threshold": True,
            "prompt": "Goblin Slayer fighting scene",
            "technical_quality": {
                "resolution": {"score": 90},
                "sharpness": {"score": 80},
                "encoding_quality": {"score": 85}
            },
            "timestamp": datetime.now().isoformat()
        },
        {
            "video_path": "/test/video2.mp4",
            "overall_score": 72.1,
            "pass_threshold": True,
            "prompt": "Magical battle scene",
            "technical_quality": {
                "resolution": {"score": 75},
                "sharpness": {"score": 70},
                "encoding_quality": {"score": 71}
            },
            "timestamp": datetime.now().isoformat()
        }
    ]

    # Add sample assessments
    for assessment in sample_assessments:
        reporter.add_assessment(assessment)

    # Generate comprehensive report
    print("📊 Generating comprehensive quality report...")
    report = reporter.generate_comprehensive_report()

    print(f"\n📋 QUALITY REPORT SUMMARY:")
    print(f"   Total Videos: {report['summary']['total_videos']}")
    print(f"   Average Score: {report['summary']['average_score']}/100")
    print(f"   Quality Grade: {report['summary']['quality_grade']}")

    # Export report
    filename = reporter.export_report_to_file(report)
    print(f"📄 Full report exported to: {filename}")

    return report

if __name__ == "__main__":
    test_quality_reporter()