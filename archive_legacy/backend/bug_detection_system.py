#!/usr/bin/env python3
"""
Bug Detection and Automated Fix Verification System
Monitors browser console, network requests, and application behavior for issues
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import re
import time

logger = logging.getLogger(__name__)

class BugSeverity(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class BugCategory(Enum):
    JAVASCRIPT_ERROR = "JavaScript Error"
    NETWORK_ERROR = "Network Error"
    CONSOLE_WARNING = "Console Warning"
    PERFORMANCE_ISSUE = "Performance Issue"
    ACCESSIBILITY_ISSUE = "Accessibility Issue"
    FUNCTIONAL_BUG = "Functional Bug"
    VISUAL_BUG = "Visual Bug"
    WEBSOCKET_ERROR = "WebSocket Error"
    API_ERROR = "API Error"

@dataclass
class Bug:
    """Represents a detected bug or issue"""
    id: str
    category: BugCategory
    severity: BugSeverity
    title: str
    description: str
    location: str
    stack_trace: str = ""
    reproduction_steps: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    actual_behavior: str = ""
    screenshot_path: str = ""
    video_path: str = ""
    network_logs: List[Dict] = field(default_factory=list)
    console_logs: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    fixed: bool = False
    fix_verification_attempts: int = 0
    max_fix_attempts: int = 3

@dataclass
class BugReport:
    """Collection of bugs found during testing"""
    test_session_id: str
    timestamp: datetime
    bugs: List[Bug] = field(default_factory=list)
    
    @property
    def critical_bugs(self) -> List[Bug]:
        return [b for b in self.bugs if b.severity == BugSeverity.CRITICAL]
    
    @property
    def high_priority_bugs(self) -> List[Bug]:
        return [b for b in self.bugs if b.severity in [BugSeverity.CRITICAL, BugSeverity.HIGH]]
    
    @property
    def unfixed_bugs(self) -> List[Bug]:
        return [b for b in self.bugs if not b.fixed]
    
    @property
    def bug_count_by_category(self) -> Dict[str, int]:
        categories = {}
        for bug in self.bugs:
            category = bug.category.value
            categories[category] = categories.get(category, 0) + 1
        return categories

class BugDetectionSystem:
    """System for detecting, tracking, and verifying fixes for bugs"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.bugs_dir = results_dir / "bugs"
        self.bugs_dir.mkdir(parents=True, exist_ok=True)
        
        # Bug detection patterns
        self.error_patterns = {
            BugCategory.JAVASCRIPT_ERROR: [
                r"TypeError:",
                r"ReferenceError:",
                r"SyntaxError:",
                r"RangeError:",
                r"Uncaught",
                r"Cannot read property",
                r"undefined is not a function"
            ],
            BugCategory.NETWORK_ERROR: [
                r"Failed to fetch",
                r"Network Error",
                r"CORS error",
                r"404 Not Found",
                r"500 Internal Server Error",
                r"Connection refused",
                r"timeout"
            ],
            BugCategory.WEBSOCKET_ERROR: [
                r"WebSocket connection failed",
                r"WebSocket error",
                r"Connection lost",
                r"WebSocket closed unexpectedly"
            ]
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "page_load_time": 5000,  # ms
            "response_time": 15000,  # ms
            "memory_usage": 500,     # MB
            "dom_size": 2000         # elements
        }
        
        # Current bug tracking
        self.current_bugs: List[Bug] = []
        self.console_logs: List[Dict] = []
        self.network_logs: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    async def initialize_monitoring(self, page):
        """Initialize browser monitoring for bug detection"""
        self.page = page
        
        # Console monitoring
        page.on("console", self._handle_console_message)
        page.on("pageerror", self._handle_page_error)
        
        # Network monitoring
        page.on("request", self._handle_request)
        page.on("response", self._handle_response)
        page.on("requestfailed", self._handle_request_failed)
        
        logger.info("Bug detection monitoring initialized")
    
    def _handle_console_message(self, msg):
        """Handle browser console messages"""
        console_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": msg.type,
            "text": msg.text,
            "location": msg.location if hasattr(msg, 'location') else ""
        }
        
        self.console_logs.append(console_entry)
        
        # Check for error patterns
        if msg.type in ["error", "warning"]:
            self._analyze_console_message(msg)
    
    def _handle_page_error(self, error):
        """Handle JavaScript page errors"""
        bug = Bug(
            id=f"JS_ERROR_{int(time.time())}",
            category=BugCategory.JAVASCRIPT_ERROR,
            severity=BugSeverity.HIGH,
            title="JavaScript Error Detected",
            description=str(error),
            location=self.page.url,
            stack_trace=str(error),
            actual_behavior=f"JavaScript error occurred: {error}",
            expected_behavior="No JavaScript errors should occur during normal operation"
        )
        
        self.current_bugs.append(bug)
        logger.error(f"JavaScript error detected: {error}")
    
    def _handle_request(self, request):
        """Handle network requests"""
        request_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "url": request.url,
            "headers": dict(request.headers),
            "type": "request"
        }
        
        self.network_logs.append(request_entry)
    
    def _handle_response(self, response):
        """Handle network responses"""
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "url": response.url,
            "status": response.status,
            "status_text": response.status_text,
            "headers": dict(response.headers),
            "type": "response"
        }
        
        self.network_logs.append(response_entry)
        
        # Check for error status codes
        if response.status >= 400:
            self._analyze_failed_response(response)
    
    def _handle_request_failed(self, request):
        """Handle failed network requests"""
        bug = Bug(
            id=f"NETWORK_ERROR_{int(time.time())}",
            category=BugCategory.NETWORK_ERROR,
            severity=BugSeverity.HIGH,
            title="Network Request Failed",
            description=f"Failed to load: {request.url}",
            location=request.url,
            actual_behavior=f"Request to {request.url} failed",
            expected_behavior="All network requests should complete successfully"
        )
        
        self.current_bugs.append(bug)
        logger.error(f"Network request failed: {request.url}")
    
    def _analyze_console_message(self, msg):
        """Analyze console message for bug patterns"""
        text = msg.text.lower()
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern.lower(), text):
                    severity = BugSeverity.HIGH if msg.type == "error" else BugSeverity.MEDIUM
                    
                    bug = Bug(
                        id=f"CONSOLE_{category.name}_{int(time.time())}",
                        category=category,
                        severity=severity,
                        title=f"Console {msg.type.title()}: {pattern}",
                        description=msg.text,
                        location=self.page.url,
                        actual_behavior=f"Console {msg.type}: {msg.text}",
                        expected_behavior=f"No console {msg.type}s should occur"
                    )
                    
                    self.current_bugs.append(bug)
                    logger.warning(f"Bug pattern detected: {pattern} in {msg.text}")
                    break
    
    def _analyze_failed_response(self, response):
        """Analyze failed HTTP responses"""
        if response.status == 404:
            severity = BugSeverity.MEDIUM
            title = "Resource Not Found (404)"
        elif response.status >= 500:
            severity = BugSeverity.HIGH
            title = "Server Error"
        else:
            severity = BugSeverity.MEDIUM
            title = f"HTTP Error {response.status}"
        
        bug = Bug(
            id=f"HTTP_ERROR_{response.status}_{int(time.time())}",
            category=BugCategory.API_ERROR,
            severity=severity,
            title=title,
            description=f"HTTP {response.status} {response.status_text} for {response.url}",
            location=response.url,
            actual_behavior=f"Received HTTP {response.status} {response.status_text}",
            expected_behavior="HTTP requests should return successful status codes"
        )
        
        self.current_bugs.append(bug)
    
    async def check_performance_issues(self):
        """Check for performance-related issues"""
        try:
            # Get performance metrics
            performance_data = await self.page.evaluate("""
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    const memory = performance.memory || {};
                    
                    return {
                        loadTime: navigation ? navigation.loadEventEnd - navigation.fetchStart : 0,
                        domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.fetchStart : 0,
                        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
                        memoryUsed: memory.usedJSHeapSize || 0,
                        memoryTotal: memory.totalJSHeapSize || 0,
                        domElements: document.querySelectorAll('*').length
                    };
                }
            """)
            
            self.performance_metrics.update(performance_data)
            
            # Check performance thresholds
            if performance_data.get('loadTime', 0) > self.performance_thresholds['page_load_time']:
                bug = Bug(
                    id=f"PERF_LOAD_TIME_{int(time.time())}",
                    category=BugCategory.PERFORMANCE_ISSUE,
                    severity=BugSeverity.MEDIUM,
                    title="Slow Page Load Time",
                    description=f"Page load time: {performance_data['loadTime']}ms exceeds threshold",
                    location=self.page.url,
                    actual_behavior=f"Page loaded in {performance_data['loadTime']}ms",
                    expected_behavior=f"Page should load within {self.performance_thresholds['page_load_time']}ms"
                )
                self.current_bugs.append(bug)
            
            memory_mb = performance_data.get('memoryUsed', 0) / (1024 * 1024)
            if memory_mb > self.performance_thresholds['memory_usage']:
                bug = Bug(
                    id=f"PERF_MEMORY_{int(time.time())}",
                    category=BugCategory.PERFORMANCE_ISSUE,
                    severity=BugSeverity.LOW,
                    title="High Memory Usage",
                    description=f"Memory usage: {memory_mb:.1f}MB exceeds threshold",
                    location=self.page.url,
                    actual_behavior=f"Using {memory_mb:.1f}MB of memory",
                    expected_behavior=f"Should use less than {self.performance_thresholds['memory_usage']}MB"
                )
                self.current_bugs.append(bug)
                
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
    
    async def check_accessibility_issues(self):
        """Check for basic accessibility issues"""
        try:
            accessibility_data = await self.page.evaluate("""
                () => {
                    const issues = [];
                    
                    // Check for images without alt text
                    const images = document.querySelectorAll('img:not([alt])');
                    if (images.length > 0) {
                        issues.push({
                            type: 'missing_alt_text',
                            count: images.length,
                            description: `${images.length} images without alt text`
                        });
                    }
                    
                    // Check for buttons without labels
                    const unlabeledButtons = document.querySelectorAll('button:not([aria-label]):not([title])');
                    if (unlabeledButtons.length > 0) {
                        issues.push({
                            type: 'unlabeled_buttons',
                            count: unlabeledButtons.length,
                            description: `${unlabeledButtons.length} buttons without labels`
                        });
                    }
                    
                    // Check for form inputs without labels
                    const unlabeledInputs = document.querySelectorAll('input:not([aria-label]):not([title])');
                    if (unlabeledInputs.length > 0) {
                        issues.push({
                            type: 'unlabeled_inputs',
                            count: unlabeledInputs.length,
                            description: `${unlabeledInputs.length} inputs without labels`
                        });
                    }
                    
                    return issues;
                }
            """)
            
            for issue in accessibility_data:
                bug = Bug(
                    id=f"A11Y_{issue['type'].upper()}_{int(time.time())}",
                    category=BugCategory.ACCESSIBILITY_ISSUE,
                    severity=BugSeverity.LOW,
                    title=f"Accessibility Issue: {issue['type']}",
                    description=issue['description'],
                    location=self.page.url,
                    actual_behavior=issue['description'],
                    expected_behavior="All interactive elements should have proper labels"
                )
                self.current_bugs.append(bug)
                
        except Exception as e:
            logger.error(f"Accessibility check failed: {e}")
    
    async def check_bahai_interface_specific_issues(self):
        """Check for Baha'i interface specific issues"""
        try:
            interface_data = await self.page.evaluate("""
                () => {
                    const issues = [];
                    
                    // Check Persian title
                    const persianTitle = document.querySelector('.title-persian');
                    if (!persianTitle) {
                        issues.push({
                            type: 'missing_persian_title',
                            description: 'Persian title element not found'
                        });
                    } else if (!persianTitle.textContent.includes('كلمات مخفیه')) {
                        issues.push({
                            type: 'incorrect_persian_title',
                            description: 'Persian title text is incorrect',
                            actual: persianTitle.textContent,
                            expected: 'كلمات مخفیه'
                        });
                    }
                    
                    // Check starfield
                    const starfield = document.querySelector('.starfield');
                    const stars = document.querySelectorAll('.star');
                    if (!starfield) {
                        issues.push({
                            type: 'missing_starfield',
                            description: 'Starfield container not found'
                        });
                    } else if (stars.length < 100) {
                        issues.push({
                            type: 'insufficient_stars',
                            description: `Only ${stars.length} stars found, expected at least 100`
                        });
                    }
                    
                    // Check WebSocket connection status
                    const status = document.querySelector('.status');
                    if (status && status.textContent.toLowerCase().includes('connection lost')) {
                        issues.push({
                            type: 'websocket_disconnected',
                            description: 'WebSocket connection appears to be lost'
                        });
                    }
                    
                    // Check input elements
                    const searchInput = document.querySelector('.search-input');
                    const revealButton = document.querySelector('.reveal-button');
                    const voiceButton = document.querySelector('.voice-button');
                    
                    if (!searchInput) {
                        issues.push({
                            type: 'missing_search_input',
                            description: 'Search input element not found'
                        });
                    }
                    
                    if (!revealButton) {
                        issues.push({
                            type: 'missing_reveal_button',
                            description: 'REVEAL button not found'
                        });
                    }
                    
                    if (!voiceButton) {
                        issues.push({
                            type: 'missing_voice_button',
                            description: 'Voice button not found'
                        });
                    }
                    
                    return issues;
                }
            """)
            
            for issue in interface_data:
                severity = BugSeverity.HIGH if 'missing' in issue['type'] else BugSeverity.MEDIUM
                
                bug = Bug(
                    id=f"BAHAI_INTERFACE_{issue['type'].upper()}_{int(time.time())}",
                    category=BugCategory.FUNCTIONAL_BUG,
                    severity=severity,
                    title=f"Interface Issue: {issue['type']}",
                    description=issue['description'],
                    location=self.page.url,
                    actual_behavior=issue.get('actual', issue['description']),
                    expected_behavior=issue.get('expected', "Interface element should be present and functional")
                )
                self.current_bugs.append(bug)
                
        except Exception as e:
            logger.error(f"Baha'i interface check failed: {e}")
    
    async def run_comprehensive_bug_scan(self) -> BugReport:
        """Run comprehensive bug detection scan"""
        logger.info("Starting comprehensive bug detection scan")
        
        # Clear previous bugs for this scan
        self.current_bugs.clear()
        
        # Run all checks
        await self.check_performance_issues()
        await self.check_accessibility_issues()
        await self.check_bahai_interface_specific_issues()
        
        # Create bug report
        report = BugReport(
            test_session_id=f"scan_{int(time.time())}",
            timestamp=datetime.now(),
            bugs=self.current_bugs.copy()
        )
        
        # Save bug report
        report_file = self.bugs_dir / f"bug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_bug_report(report, report_file)
        
        logger.info(f"Bug scan completed. Found {len(report.bugs)} issues. Report saved to: {report_file}")
        return report
    
    async def verify_bug_fixes(self, bug_report: BugReport) -> Dict[str, Any]:
        """Verify that previously detected bugs have been fixed"""
        logger.info("Starting bug fix verification")
        
        verification_results = {
            "verified_fixes": [],
            "persistent_bugs": [],
            "new_bugs": [],
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Re-run checks to see current state
        current_scan = await self.run_comprehensive_bug_scan()
        
        # Compare with previous bug report
        previous_bug_ids = {bug.id for bug in bug_report.bugs}
        current_bug_ids = {bug.id for bug in current_scan.bugs}
        
        # Find fixed bugs (in previous but not in current)
        fixed_bug_ids = previous_bug_ids - current_bug_ids
        verification_results["verified_fixes"] = list(fixed_bug_ids)
        
        # Find persistent bugs (in both)
        persistent_bug_ids = previous_bug_ids & current_bug_ids
        verification_results["persistent_bugs"] = list(persistent_bug_ids)
        
        # Find new bugs (in current but not in previous)
        new_bug_ids = current_bug_ids - previous_bug_ids
        verification_results["new_bugs"] = list(new_bug_ids)
        
        # Update fix status
        for bug in bug_report.bugs:
            if bug.id in fixed_bug_ids:
                bug.fixed = True
            elif bug.id in persistent_bug_ids:
                bug.fix_verification_attempts += 1
        
        # Save updated bug report
        report_file = self.bugs_dir / f"bug_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        verification_data = {
            "original_report": self._serialize_bug_report(bug_report),
            "current_scan": self._serialize_bug_report(current_scan),
            "verification_results": verification_results
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(verification_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(
            f"Fix verification completed. "
            f"Fixed: {len(verification_results['verified_fixes'])}, "
            f"Persistent: {len(verification_results['persistent_bugs'])}, "
            f"New: {len(verification_results['new_bugs'])}"
        )
        
        return verification_results
    
    def save_bug_report(self, report: BugReport, file_path: Path):
        """Save bug report to JSON file"""
        serialized_report = self._serialize_bug_report(report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_report, f, indent=2, ensure_ascii=False, default=str)
    
    def _serialize_bug_report(self, report: BugReport) -> Dict[str, Any]:
        """Convert bug report to JSON-serializable format"""
        return {
            "test_session_id": report.test_session_id,
            "timestamp": report.timestamp.isoformat(),
            "summary": {
                "total_bugs": len(report.bugs),
                "critical_bugs": len(report.critical_bugs),
                "high_priority_bugs": len(report.high_priority_bugs),
                "unfixed_bugs": len(report.unfixed_bugs),
                "bug_count_by_category": report.bug_count_by_category
            },
            "bugs": [
                {
                    "id": bug.id,
                    "category": bug.category.value,
                    "severity": bug.severity.name,
                    "title": bug.title,
                    "description": bug.description,
                    "location": bug.location,
                    "stack_trace": bug.stack_trace,
                    "reproduction_steps": bug.reproduction_steps,
                    "expected_behavior": bug.expected_behavior,
                    "actual_behavior": bug.actual_behavior,
                    "screenshot_path": bug.screenshot_path,
                    "video_path": bug.video_path,
                    "timestamp": bug.timestamp.isoformat(),
                    "fixed": bug.fixed,
                    "fix_verification_attempts": bug.fix_verification_attempts
                }
                for bug in report.bugs
            ],
            "console_logs": self.console_logs[-100:],  # Last 100 entries
            "network_logs": self.network_logs[-50:],   # Last 50 entries
            "performance_metrics": self.performance_metrics
        }
    
    def generate_bug_summary_report(self, reports: List[BugReport]) -> Dict[str, Any]:
        """Generate summary report across multiple bug reports"""
        if not reports:
            return {"error": "No bug reports provided"}
        
        all_bugs = []
        for report in reports:
            all_bugs.extend(report.bugs)
        
        # Calculate trends
        fixed_bugs = [bug for bug in all_bugs if bug.fixed]
        unfixed_bugs = [bug for bug in all_bugs if not bug.fixed]
        
        category_trends = {}
        severity_trends = {}
        
        for bug in all_bugs:
            category = bug.category.value
            severity = bug.severity.name
            
            category_trends[category] = category_trends.get(category, 0) + 1
            severity_trends[severity] = severity_trends.get(severity, 0) + 1
        
        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_reports_analyzed": len(reports),
            "total_bugs_found": len(all_bugs),
            "fixed_bugs": len(fixed_bugs),
            "unfixed_bugs": len(unfixed_bugs),
            "fix_rate": (len(fixed_bugs) / len(all_bugs) * 100) if all_bugs else 0,
            "category_distribution": category_trends,
            "severity_distribution": severity_trends,
            "most_common_issues": self._get_most_common_issues(all_bugs),
            "recommendations": self._generate_bug_recommendations(all_bugs)
        }
        
        # Save summary
        summary_file = self.bugs_dir / f"bug_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def _get_most_common_issues(self, bugs: List[Bug]) -> List[Dict[str, Any]]:
        """Identify most common issue patterns"""
        issue_patterns = {}
        
        for bug in bugs:
            # Group by category and title similarity
            key = f"{bug.category.value}_{bug.title[:50]}"
            if key not in issue_patterns:
                issue_patterns[key] = {
                    "pattern": bug.title,
                    "category": bug.category.value,
                    "count": 0,
                    "severity": bug.severity.name
                }
            issue_patterns[key]["count"] += 1
        
        # Sort by frequency
        common_issues = sorted(
            issue_patterns.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return common_issues[:10]  # Top 10 most common
    
    def _generate_bug_recommendations(self, bugs: List[Bug]) -> List[str]:
        """Generate recommendations based on bug analysis"""
        recommendations = []
        
        js_errors = [b for b in bugs if b.category == BugCategory.JAVASCRIPT_ERROR]
        if js_errors:
            recommendations.append(
                f"Found {len(js_errors)} JavaScript errors. "
                "Implement error boundaries and improve error handling."
            )
        
        network_errors = [b for b in bugs if b.category == BugCategory.NETWORK_ERROR]
        if network_errors:
            recommendations.append(
                f"Found {len(network_errors)} network errors. "
                "Implement retry logic and better error messaging."
            )
        
        websocket_errors = [b for b in bugs if b.category == BugCategory.WEBSOCKET_ERROR]
        if websocket_errors:
            recommendations.append(
                f"Found {len(websocket_errors)} WebSocket errors. "
                "Improve connection stability and reconnection logic."
            )
        
        perf_issues = [b for b in bugs if b.category == BugCategory.PERFORMANCE_ISSUE]
        if perf_issues:
            recommendations.append(
                f"Found {len(perf_issues)} performance issues. "
                "Optimize loading times and resource usage."
            )
        
        critical_bugs = [b for b in bugs if b.severity == BugSeverity.CRITICAL]
        if critical_bugs:
            recommendations.append(
                f"Found {len(critical_bugs)} critical bugs that need immediate attention."
            )
        
        unfixed_bugs = [b for b in bugs if not b.fixed and b.fix_verification_attempts > 0]
        if unfixed_bugs:
            recommendations.append(
                f"Found {len(unfixed_bugs)} persistent bugs that have not been fixed after attempts."
            )
        
        return recommendations