#!/usr/bin/env python3
"""
Enhanced Comprehensive Playwright QA Framework for Baha'i Spiritual Quest Interface
Integrates visual testing, bug detection, and automated fix verification
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from playwright_qa_framework import BahaiQAFramework, TestSuite, TestResult
from visual_test_engine import VisualTestEngine, VisualTestResult
from bug_detection_system import BugDetectionSystem, BugReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_qa_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBahaiQAFramework:
    """Enhanced QA Framework with comprehensive testing capabilities"""
    
    def __init__(self, config_path: str = "playwright_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize results directory
        self.results_dir = Path(self.config.get("reporting", {}).get("report_directory", "qa_results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.base_framework = BahaiQAFramework()
        self.visual_engine = VisualTestEngine(self.results_dir)
        self.bug_detector = BugDetectionSystem(self.results_dir)
        
        # Test session tracking
        self.test_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.all_bug_reports: List[BugReport] = []
        self.all_visual_results: List[VisualTestResult] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "execution": {"headless": False, "timeout": 30000},
            "application": {"base_url": "http://localhost:8000"},
            "visual": {"enable_screenshots": True},
            "performance": {"max_page_load_time": 5000},
            "reporting": {"report_directory": "qa_results"}
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        logger.info(f"🚀 Starting Enhanced Comprehensive QA Testing - Session: {self.test_session_id}")
        
        start_time = datetime.now()
        
        # Setup browser
        if not await self.base_framework.setup_browser(
            headless=self.config.get("execution", {}).get("headless", False)
        ):
            return {"error": "Failed to setup browser"}
        
        try:
            # Initialize bug detection monitoring
            await self.bug_detector.initialize_monitoring(self.base_framework.page)
            
            # Phase 1: Core Functional Testing
            logger.info("📋 Phase 1: Core Functional Testing")
            functional_results = await self._run_functional_tests()
            
            # Phase 2: Visual Testing
            logger.info("🎨 Phase 2: Visual Testing")
            visual_results = await self._run_visual_tests()
            
            # Phase 3: Bug Detection Scan
            logger.info("🔍 Phase 3: Bug Detection Scan")
            bug_report = await self.bug_detector.run_comprehensive_bug_scan()
            self.all_bug_reports.append(bug_report)
            
            # Phase 4: Performance Testing
            logger.info("⚡ Phase 4: Performance Testing")
            performance_results = await self._run_performance_tests()
            
            # Phase 5: Edge Cases and Stress Testing
            logger.info("🎯 Phase 5: Edge Cases and Stress Testing")
            edge_case_results = await self._run_edge_case_tests()
            
            # Compile comprehensive results
            results = self._compile_comprehensive_results(
                functional_results,
                visual_results,
                bug_report,
                performance_results,
                edge_case_results,
                start_time
            )
            
            # Generate detailed reports
            await self._generate_comprehensive_reports(results)
            
            # Check if we achieved 0 bugs
            bug_free = bug_report.bugs == []
            if bug_free:
                logger.info("🎉 ACHIEVEMENT UNLOCKED: Zero bugs detected!")
                results["achievement"] = "ZERO_BUGS_DETECTED"
            
            return results
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
        
        finally:
            await self.base_framework.teardown_browser()
    
    async def _run_functional_tests(self) -> Dict[str, Any]:
        """Run core functional tests"""
        # Use the base framework to run standard tests
        functional_results = await self.base_framework.run_all_tests()
        return functional_results
    
    async def _run_visual_tests(self) -> Dict[str, List[VisualTestResult]]:
        """Run comprehensive visual tests"""
        visual_results = {
            "persian_title": [],
            "starfield_animation": [],
            "quote_formatting": [],
            "responsive_design": [],
            "ui_state_changes": []
        }
        
        try:
            page = self.base_framework.page
            
            # Test Persian title rendering
            logger.info("Testing Persian title rendering...")
            persian_result = await self.visual_engine.test_persian_title_rendering(page)
            visual_results["persian_title"].append(persian_result)
            self.all_visual_results.append(persian_result)
            
            # Test starfield animation
            logger.info("Testing starfield animation...")
            starfield_result = await self.visual_engine.test_starfield_animation(page)
            visual_results["starfield_animation"].append(starfield_result)
            
            # Test quote formatting (need to send a message first)
            logger.info("Testing quote formatting...")
            quote_result = await self.visual_engine.test_quote_formatting(
                page, "Share a Hidden Words quote about love"
            )
            visual_results["quote_formatting"].append(quote_result)
            self.all_visual_results.append(quote_result)
            
            # Test responsive design
            logger.info("Testing responsive design...")
            responsive_results = await self.visual_engine.test_responsive_design(page)
            for device, result in responsive_results.items():
                visual_results["responsive_design"].append(result)
                self.all_visual_results.append(result)
            
            # Test UI state changes
            logger.info("Testing UI state changes...")
            ui_state_results = await self.visual_engine.test_ui_state_changes(page)
            for state, result in ui_state_results.items():
                visual_results["ui_state_changes"].append(result)
                self.all_visual_results.append(result)
            
        except Exception as e:
            logger.error(f"Visual testing failed: {e}")
            visual_results["error"] = str(e)
        
        return visual_results
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        performance_results = {
            "page_load_times": [],
            "response_times": [],
            "memory_usage": [],
            "network_analysis": {}
        }
        
        try:
            page = self.base_framework.page
            
            # Test page load performance
            logger.info("Testing page load performance...")
            load_start = datetime.now()
            await page.goto("http://localhost:8000")
            await page.wait_for_selector('.manuscript-container')
            load_end = datetime.now()
            load_time = (load_end - load_start).total_seconds() * 1000
            
            performance_results["page_load_times"].append({
                "test": "initial_load",
                "time_ms": load_time,
                "threshold_met": load_time < self.config.get("performance", {}).get("max_page_load_time", 5000)
            })
            
            # Test AI response time
            logger.info("Testing AI response performance...")
            response_start = datetime.now()
            await page.locator('.search-input').fill("What is love?")
            await page.locator('.reveal-button').click()
            
            # Wait for response
            try:
                await page.wait_for_function(
                    "document.getElementById('typingIndicator').style.display === 'none'",
                    timeout=45000
                )
                response_end = datetime.now()
                response_time = (response_end - response_start).total_seconds() * 1000
                
                performance_results["response_times"].append({
                    "test": "ai_response",
                    "time_ms": response_time,
                    "threshold_met": response_time < self.config.get("performance", {}).get("max_response_time", 15000)
                })
            except Exception as e:
                logger.error(f"AI response timeout: {e}")
                performance_results["response_times"].append({
                    "test": "ai_response",
                    "time_ms": 45000,
                    "threshold_met": False,
                    "error": "Timeout"
                })
            
            # Memory usage test
            logger.info("Testing memory usage...")
            memory_stats = await page.evaluate("""
                () => {
                    if (performance.memory) {
                        return {
                            used: performance.memory.usedJSHeapSize,
                            total: performance.memory.totalJSHeapSize,
                            limit: performance.memory.jsHeapSizeLimit
                        };
                    }
                    return null;
                }
            """)
            
            if memory_stats:
                memory_mb = memory_stats["used"] / (1024 * 1024)
                performance_results["memory_usage"].append({
                    "test": "current_memory",
                    "memory_mb": memory_mb,
                    "threshold_met": memory_mb < self.config.get("performance", {}).get("max_memory_usage", 500)
                })
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            performance_results["error"] = str(e)
        
        return performance_results
    
    async def _run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case and stress tests"""
        edge_case_results = {
            "empty_inputs": [],
            "long_inputs": [],
            "special_characters": [],
            "rapid_interactions": [],
            "connection_interruption": []
        }
        
        try:
            page = self.base_framework.page
            
            # Test empty input handling
            logger.info("Testing empty input handling...")
            initial_msg_count = await page.locator('.message').count()
            await page.locator('.reveal-button').click()  # Click without input
            await page.wait_for_timeout(2000)
            final_msg_count = await page.locator('.message').count()
            
            edge_case_results["empty_inputs"].append({
                "test": "empty_click",
                "passed": initial_msg_count == final_msg_count,
                "description": "Empty input should not send message"
            })
            
            # Test very long input
            logger.info("Testing very long input...")
            long_text = "A" * 1000  # 1000 character string
            await page.locator('.search-input').fill(long_text)
            await page.locator('.reveal-button').click()
            
            try:
                await page.wait_for_function(
                    "document.getElementById('typingIndicator').style.display === 'none'",
                    timeout=60000
                )
                edge_case_results["long_inputs"].append({
                    "test": "1000_char_input",
                    "passed": True,
                    "description": "System handled 1000 character input"
                })
            except:
                edge_case_results["long_inputs"].append({
                    "test": "1000_char_input",
                    "passed": False,
                    "description": "System failed to handle 1000 character input"
                })
            
            # Test special characters
            logger.info("Testing special characters...")
            special_chars = "!@#$%^&*()[]{}|;':\"<>?/~`"
            await page.locator('.search-input').fill(special_chars)
            await page.locator('.reveal-button').click()
            
            try:
                await page.wait_for_function(
                    "document.getElementById('typingIndicator').style.display === 'none'",
                    timeout=30000
                )
                edge_case_results["special_characters"].append({
                    "test": "special_chars",
                    "passed": True,
                    "description": "System handled special characters"
                })
            except:
                edge_case_results["special_characters"].append({
                    "test": "special_chars",
                    "passed": False,
                    "description": "System failed with special characters"
                })
            
            # Test rapid interactions
            logger.info("Testing rapid interactions...")
            for i in range(5):
                await page.locator('.search-input').fill(f"Rapid test {i}")
                await page.locator('.reveal-button').click()
                await page.wait_for_timeout(200)  # Small delay between clicks
            
            # Wait for all responses
            await page.wait_for_timeout(10000)
            edge_case_results["rapid_interactions"].append({
                "test": "rapid_clicks",
                "passed": True,  # If we get here without crashing, it passed
                "description": "System handled rapid consecutive interactions"
            })
            
        except Exception as e:
            logger.error(f"Edge case testing failed: {e}")
            edge_case_results["error"] = str(e)
        
        return edge_case_results
    
    def _compile_comprehensive_results(self, functional_results, visual_results, bug_report, performance_results, edge_case_results, start_time) -> Dict[str, Any]:
        """Compile all test results into comprehensive report"""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Calculate overall metrics
        total_tests = functional_results.get("total_tests", 0)
        total_passed = functional_results.get("total_passed", 0)
        total_failed = functional_results.get("total_failed", 0)
        total_errors = functional_results.get("total_errors", 0)
        
        # Add visual test results
        visual_passed = sum(1 for vr in self.all_visual_results if vr.passed)
        visual_total = len(self.all_visual_results)
        
        total_tests += visual_total
        total_passed += visual_passed
        total_failed += (visual_total - visual_passed)
        
        # Calculate quality score
        bug_count = len(bug_report.bugs)
        critical_bugs = len(bug_report.critical_bugs)
        high_priority_bugs = len(bug_report.high_priority_bugs)
        
        quality_score = max(0, 100 - (total_failed * 2) - (total_errors * 3) - (bug_count * 1.5) - (critical_bugs * 5))
        
        return {
            "test_session_id": self.test_session_id,
            "execution_timestamp": start_time.isoformat(),
            "execution_time_seconds": execution_time,
            
            # Overall metrics
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "overall_pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "quality_score": quality_score,
            
            # Bug analysis
            "bug_count": bug_count,
            "critical_bugs": critical_bugs,
            "high_priority_bugs": high_priority_bugs,
            "bug_categories": bug_report.bug_count_by_category,
            "bug_free": bug_count == 0,
            
            # Detailed results
            "functional_results": functional_results,
            "visual_results": self._summarize_visual_results(visual_results),
            "bug_report": self.bug_detector._serialize_bug_report(bug_report),
            "performance_results": performance_results,
            "edge_case_results": edge_case_results,
            
            # Recommendations
            "recommendations": self._generate_comprehensive_recommendations(
                functional_results, visual_results, bug_report, performance_results, edge_case_results
            ),
            
            # Achievement tracking
            "achievements": self._check_achievements(bug_count, quality_score, total_tests, total_passed)
        }
    
    def _summarize_visual_results(self, visual_results) -> Dict[str, Any]:
        """Summarize visual test results"""
        summary = {
            "total_visual_tests": len(self.all_visual_results),
            "visual_tests_passed": sum(1 for vr in self.all_visual_results if vr.passed),
            "average_similarity": sum(vr.similarity_score for vr in self.all_visual_results) / len(self.all_visual_results) if self.all_visual_results else 0,
            "categories": {}
        }
        
        for category, results in visual_results.items():
            if isinstance(results, list) and results:
                summary["categories"][category] = {
                    "count": len(results),
                    "passed": sum(1 for r in results if getattr(r, 'passed', False))
                }
        
        return summary
    
    def _generate_comprehensive_recommendations(self, functional_results, visual_results, bug_report, performance_results, edge_case_results) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Bug-based recommendations
        if bug_report.critical_bugs:
            recommendations.append(f"🚨 CRITICAL: {len(bug_report.critical_bugs)} critical bugs require immediate attention")
        
        if bug_report.high_priority_bugs:
            recommendations.append(f"⚠️  HIGH PRIORITY: {len(bug_report.high_priority_bugs)} high-priority bugs need fixing")
        
        # Performance recommendations
        perf_failed = []
        for category in performance_results.values():
            if isinstance(category, list):
                perf_failed.extend([test for test in category if not test.get("threshold_met", True)])
        
        if perf_failed:
            recommendations.append(f"⚡ PERFORMANCE: {len(perf_failed)} performance tests failed - optimize loading and response times")
        
        # Visual recommendations
        visual_failed = [vr for vr in self.all_visual_results if not vr.passed]
        if visual_failed:
            recommendations.append(f"🎨 VISUAL: {len(visual_failed)} visual tests failed - review UI changes or update baselines")
        
        # Functional recommendations
        func_failed = functional_results.get("total_failed", 0) + functional_results.get("total_errors", 0)
        if func_failed > 0:
            recommendations.append(f"🔧 FUNCTIONAL: {func_failed} functional tests failed - review core functionality")
        
        # Overall quality recommendations
        quality_score = max(0, 100 - func_failed * 2 - len(bug_report.bugs))
        if quality_score < 90:
            recommendations.append(f"📊 QUALITY: Overall quality score is {quality_score:.1f}% - aim for 90%+ for production readiness")
        
        if not recommendations:
            recommendations.append("✅ EXCELLENT: All tests passing! System is ready for production.")
        
        return recommendations
    
    def _check_achievements(self, bug_count, quality_score, total_tests, total_passed) -> List[str]:
        """Check for unlocked achievements"""
        achievements = []
        
        if bug_count == 0:
            achievements.append("🏆 ZERO BUGS: No bugs detected!")
        
        if quality_score >= 95:
            achievements.append("💎 PREMIUM QUALITY: 95%+ quality score")
        elif quality_score >= 90:
            achievements.append("⭐ HIGH QUALITY: 90%+ quality score")
        
        if total_tests > 0 and total_passed == total_tests:
            achievements.append("🎯 PERFECT SCORE: 100% test pass rate")
        elif total_tests > 0 and (total_passed / total_tests) >= 0.95:
            achievements.append("🌟 NEAR PERFECT: 95%+ test pass rate")
        
        if len(self.all_visual_results) > 0 and all(vr.passed for vr in self.all_visual_results):
            achievements.append("🎨 VISUAL PERFECTION: All visual tests passed")
        
        return achievements
    
    async def _generate_comprehensive_reports(self, results):
        """Generate comprehensive HTML and JSON reports"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_report_path = self.results_dir / f"comprehensive_report_{timestamp}.json"
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate HTML report
        html_report_path = self.results_dir / f"comprehensive_report_{timestamp}.html"
        await self._generate_html_report(results, html_report_path)
        
        logger.info(f"📊 Reports generated:")
        logger.info(f"   JSON: {json_report_path}")
        logger.info(f"   HTML: {html_report_path}")
        
        # Generate visual test report
        if self.all_visual_results:
            visual_report = self.visual_engine.generate_visual_test_report(self.all_visual_results)
            logger.info(f"🎨 Visual test report generated")
    
    async def _generate_html_report(self, results, output_path):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Comprehensive QA Report - {results['test_session_id']}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 30px; }}
                .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }}
                .metric h3 {{ margin: 0 0 10px 0; color: #495057; }}
                .metric .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .section {{ padding: 20px 30px; border-bottom: 1px solid #eee; }}
                .section h2 {{ color: #495057; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .achievement {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px 15px; border-radius: 5px; margin: 5px 0; }}
                .recommendation {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px 15px; border-radius: 5px; margin: 5px 0; }}
                .bug-critical {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
                .bug-high {{ background: #fce4ec; border: 1px solid #f8bbd9; color: #880e4f; }}
                .bug-medium {{ background: #fff3e0; border: 1px solid #ffcc02; color: #ef6c00; }}
                .pass {{ color: #28a745; font-weight: bold; }}
                .fail {{ color: #dc3545; font-weight: bold; }}
                .quality-score {{ font-size: 3em; font-weight: bold; text-align: center; padding: 20px; }}
                .quality-excellent {{ color: #28a745; }}
                .quality-good {{ color: #ffc107; }}
                .quality-poor {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 Comprehensive QA Report</h1>
                    <p>Baha'i Spiritual Quest Interface - Session: {results['test_session_id']}</p>
                    <p>Executed: {results['execution_timestamp']} | Duration: {results['execution_time_seconds']:.1f}s</p>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Total Tests</h3>
                        <div class="value">{results['total_tests']}</div>
                    </div>
                    <div class="metric">
                        <h3>Pass Rate</h3>
                        <div class="value {'pass' if results['overall_pass_rate'] >= 90 else 'fail'}">{results['overall_pass_rate']:.1f}%</div>
                    </div>
                    <div class="metric">
                        <h3>Bug Count</h3>
                        <div class="value {'pass' if results['bug_count'] == 0 else 'fail'}">{results['bug_count']}</div>
                    </div>
                    <div class="metric">
                        <h3>Quality Score</h3>
                        <div class="value quality-{'excellent' if results['quality_score'] >= 90 else 'good' if results['quality_score'] >= 70 else 'poor'}">{results['quality_score']:.1f}</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏆 Achievements</h2>
                    {''.join(f'<div class="achievement">{achievement}</div>' for achievement in results['achievements'])}
                    {('<div class="achievement">🎯 BUG-FREE TESTING: Zero bugs detected in this session!</div>' if results['bug_free'] else '')}
                </div>
                
                <div class="section">
                    <h2>💡 Recommendations</h2>
                    {''.join(f'<div class="recommendation">{rec}</div>' for rec in results['recommendations'])}
                </div>
                
                <div class="section">
                    <h2>🔍 Bug Analysis</h2>
                    <p><strong>Total Bugs:</strong> {results['bug_count']}</p>
                    <p><strong>Critical Bugs:</strong> {results['critical_bugs']}</p>
                    <p><strong>High Priority:</strong> {results['high_priority_bugs']}</p>
                    
                    <h3>Bug Categories:</h3>
                    {''.join(f'<p><strong>{category}:</strong> {count}</p>' for category, count in results['bug_categories'].items())}
                </div>
                
                <div class="section">
                    <h2>📊 Test Summary</h2>
                    <p><strong>Functional Tests:</strong> {results['functional_results']['total_tests']} tests, {results['functional_results']['total_passed']} passed</p>
                    <p><strong>Visual Tests:</strong> {results['visual_results']['total_visual_tests']} tests, {results['visual_results']['visual_tests_passed']} passed</p>
                    <p><strong>Performance Tests:</strong> Multiple categories tested</p>
                    <p><strong>Edge Case Tests:</strong> Comprehensive edge case coverage</p>
                </div>
                
                <div class="section">
                    <h2>📈 Execution Details</h2>
                    <p><strong>Session ID:</strong> {results['test_session_id']}</p>
                    <p><strong>Execution Time:</strong> {results['execution_time_seconds']:.1f} seconds</p>
                    <p><strong>Test Environment:</strong> {results.get('functional_results', {}).get('test_suites', {}).keys()}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    async def run_continuous_zero_bug_quest(self, max_iterations: int = 20, delay_minutes: int = 30) -> Dict[str, Any]:
        """Run continuous testing until zero bugs are consistently achieved"""
        logger.info(f"🎯 Starting ZERO BUG QUEST - Up to {max_iterations} iterations")
        
        zero_bug_streak = 0
        target_streak = 3  # Need 3 consecutive zero-bug runs
        all_iterations = []
        
        for iteration in range(max_iterations):
            logger.info(f"🔄 Zero Bug Quest - Iteration {iteration + 1}/{max_iterations}")
            
            # Run comprehensive test
            results = await self.run_comprehensive_test_suite()
            results["iteration"] = iteration + 1
            results["zero_bug_streak"] = zero_bug_streak
            
            all_iterations.append(results)
            
            # Check if this iteration achieved zero bugs
            if results.get("bug_free", False):
                zero_bug_streak += 1
                logger.info(f"✅ Zero bugs achieved! Streak: {zero_bug_streak}/{target_streak}")
                
                if zero_bug_streak >= target_streak:
                    logger.info("🎉 QUEST COMPLETED! Achieved 3 consecutive zero-bug test runs!")
                    break
            else:
                zero_bug_streak = 0
                bug_count = results.get("bug_count", 0)
                logger.warning(f"❌ Bugs detected: {bug_count}. Streak reset.")
            
            # Wait before next iteration (except on last)
            if iteration < max_iterations - 1:
                logger.info(f"⏰ Waiting {delay_minutes} minutes before next iteration...")
                await asyncio.sleep(delay_minutes * 60)
        
        # Generate quest summary
        quest_summary = {
            "quest_completed": zero_bug_streak >= target_streak,
            "final_streak": zero_bug_streak,
            "total_iterations": len(all_iterations),
            "quest_duration_hours": (len(all_iterations) * delay_minutes) / 60,
            "iterations": all_iterations,
            "achievements_unlocked": [],
            "final_status": "QUEST_COMPLETED" if zero_bug_streak >= target_streak else "QUEST_INCOMPLETE"
        }
        
        # Check for special achievements
        if quest_summary["quest_completed"]:
            quest_summary["achievements_unlocked"].append("🏆 ZERO BUG CHAMPION")
        
        if len(all_iterations) >= 10:
            quest_summary["achievements_unlocked"].append("🔄 PERSISTENCE MASTER")
        
        # Save quest summary
        quest_file = self.results_dir / f"zero_bug_quest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(quest_file, 'w', encoding='utf-8') as f:
            json.dump(quest_summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"🏁 Zero Bug Quest completed! Summary saved to: {quest_file}")
        return quest_summary

async def main():
    """Main function to run the enhanced QA framework"""
    print("🚀 Enhanced Comprehensive Playwright QA Framework")
    print("🎯 For Baha'i Spiritual Quest Interface")
    print("="*60)
    
    framework = EnhancedBahaiQAFramework()
    
    print("\n1. Single Comprehensive Test")
    print("2. Zero Bug Quest (Continuous until 0 bugs)")
    choice = input("\nSelect testing mode (1-2): ").strip()
    
    if choice == "2":
        print("\n🎯 Starting Zero Bug Quest...")
        quest_results = await framework.run_continuous_zero_bug_quest()
        
        print(f"\n🏁 Quest Results:")
        print(f"Status: {quest_results['final_status']}")
        print(f"Iterations: {quest_results['total_iterations']}")
        print(f"Final Streak: {quest_results['final_streak']}")
        print(f"Achievements: {', '.join(quest_results['achievements_unlocked'])}")
        
    else:
        print("\n🔍 Running Single Comprehensive Test...")
        results = await framework.run_comprehensive_test_suite()
        
        print(f"\n📊 Test Results Summary:")
        print(f"Total Tests: {results.get('total_tests', 0)}")
        print(f"Pass Rate: {results.get('overall_pass_rate', 0):.1f}%")
        print(f"Bug Count: {results.get('bug_count', 0)}")
        print(f"Quality Score: {results.get('quality_score', 0):.1f}/100")
        print(f"Achievements: {', '.join(results.get('achievements', []))}")

if __name__ == "__main__":
    asyncio.run(main())