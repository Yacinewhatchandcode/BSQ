#!/usr/bin/env python3
"""
Visual Testing Engine for Baha'i Spiritual Quest Interface
Handles screenshot comparison, animation validation, and visual regression testing
"""

import asyncio
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VisualTestResult:
    """Results from visual comparison"""
    test_name: str
    passed: bool
    similarity_score: float
    difference_percentage: float
    baseline_path: str
    current_path: str
    diff_path: str
    threshold_met: bool
    error_message: str = ""

class VisualTestEngine:
    """Engine for performing visual testing and comparisons"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.screenshots_dir = results_dir / "screenshots"
        self.baselines_dir = results_dir / "baselines"
        self.diffs_dir = results_dir / "diffs"
        
        # Create directories
        for directory in [self.screenshots_dir, self.baselines_dir, self.diffs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Visual comparison thresholds
        self.pixel_threshold = 0.2  # Maximum pixel difference (0-1)
        self.difference_threshold = 0.05  # Maximum 5% difference
        
    async def capture_element_screenshot(self, page, selector: str, filename: str) -> Path:
        """Capture screenshot of a specific element"""
        try:
            element = page.locator(selector)
            await element.wait_for(timeout=10000)
            
            screenshot_path = self.screenshots_dir / f"{filename}.png"
            await element.screenshot(path=str(screenshot_path))
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture element screenshot {filename}: {e}")
            raise

    async def capture_full_page_screenshot(self, page, filename: str) -> Path:
        """Capture full page screenshot"""
        try:
            screenshot_path = self.screenshots_dir / f"{filename}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture full page screenshot {filename}: {e}")
            raise

    def compare_images(self, baseline_path: Path, current_path: Path, test_name: str) -> VisualTestResult:
        """Compare two images and return detailed results"""
        try:
            # Load images
            baseline = cv2.imread(str(baseline_path))
            current = cv2.imread(str(current_path))
            
            if baseline is None or current is None:
                return VisualTestResult(
                    test_name=test_name,
                    passed=False,
                    similarity_score=0.0,
                    difference_percentage=100.0,
                    baseline_path=str(baseline_path),
                    current_path=str(current_path),
                    diff_path="",
                    threshold_met=False,
                    error_message="Failed to load one or both images"
                )
            
            # Resize images to match if needed
            if baseline.shape != current.shape:
                current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))
            
            # Calculate structural similarity
            gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_baseline, gray_current)
            
            # Calculate difference percentage
            non_zero_pixels = np.count_nonzero(diff)
            total_pixels = diff.shape[0] * diff.shape[1]
            difference_percentage = (non_zero_pixels / total_pixels) * 100
            
            # Calculate similarity score
            similarity_score = 100 - difference_percentage
            
            # Create difference image
            diff_path = self.diffs_dir / f"{test_name}_diff.png"
            
            # Create colored difference image
            diff_colored = np.zeros_like(baseline)
            diff_colored[:, :, 2] = diff  # Red channel for differences
            
            # Overlay difference on original
            overlay = cv2.addWeighted(baseline, 0.7, diff_colored, 0.3, 0)
            cv2.imwrite(str(diff_path), overlay)
            
            # Check if difference is within threshold
            threshold_met = difference_percentage <= self.difference_threshold * 100
            
            return VisualTestResult(
                test_name=test_name,
                passed=threshold_met,
                similarity_score=similarity_score,
                difference_percentage=difference_percentage,
                baseline_path=str(baseline_path),
                current_path=str(current_path),
                diff_path=str(diff_path),
                threshold_met=threshold_met
            )
            
        except Exception as e:
            logger.error(f"Image comparison failed for {test_name}: {e}")
            return VisualTestResult(
                test_name=test_name,
                passed=False,
                similarity_score=0.0,
                difference_percentage=100.0,
                baseline_path=str(baseline_path),
                current_path=str(current_path),
                diff_path="",
                threshold_met=False,
                error_message=str(e)
            )
    
    async def test_persian_title_rendering(self, page) -> VisualTestResult:
        """Test Persian title rendering and font display"""
        test_name = "persian_title_rendering"
        
        # Capture current Persian title
        current_path = await self.capture_element_screenshot(
            page, '.title-persian', f"{test_name}_current"
        )
        
        # Check for baseline
        baseline_path = self.baselines_dir / f"{test_name}_baseline.png"
        
        if not baseline_path.exists():
            # Create baseline on first run
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(str(current_path), str(baseline_path))
            logger.info(f"Created baseline for {test_name}")
            
            return VisualTestResult(
                test_name=test_name,
                passed=True,
                similarity_score=100.0,
                difference_percentage=0.0,
                baseline_path=str(baseline_path),
                current_path=str(current_path),
                diff_path="",
                threshold_met=True
            )
        
        return self.compare_images(baseline_path, current_path, test_name)
    
    async def test_starfield_animation(self, page) -> Dict[str, Any]:
        """Test starfield animation by capturing multiple frames"""
        test_name = "starfield_animation"
        animation_results = []
        
        try:
            # Wait for starfield to load
            await page.wait_for_selector('.starfield', timeout=10000)
            await page.wait_for_timeout(2000)  # Let animation settle
            
            # Capture multiple frames
            frames = []
            for i in range(5):
                frame_path = await self.capture_element_screenshot(
                    page, '.starfield', f"{test_name}_frame_{i}"
                )
                frames.append(frame_path)
                await page.wait_for_timeout(1000)  # 1 second between frames
            
            # Compare consecutive frames to detect animation
            animation_detected = False
            for i in range(len(frames) - 1):
                result = self.compare_images(frames[i], frames[i + 1], f"{test_name}_diff_{i}")
                if result.difference_percentage > 0.1:  # Animation threshold
                    animation_detected = True
                animation_results.append(result)
            
            # Count stars
            star_count = await page.locator('.star').count()
            
            return {
                "test_name": test_name,
                "passed": animation_detected and star_count >= 100,
                "animation_detected": animation_detected,
                "star_count": star_count,
                "frame_comparisons": animation_results,
                "frames_captured": len(frames)
            }
            
        except Exception as e:
            logger.error(f"Starfield animation test failed: {e}")
            return {
                "test_name": test_name,
                "passed": False,
                "error": str(e)
            }
    
    async def test_quote_formatting(self, page, message: str = "Share a Hidden Words quote") -> VisualTestResult:
        """Test visual formatting of Hidden Words quotes"""
        test_name = "quote_formatting"
        
        try:
            # Send message to get quote
            await page.locator('.search-input').fill(message)
            await page.locator('.reveal-button').click()
            
            # Wait for response with quote
            await page.wait_for_selector('.hidden-word-quote', timeout=45000)
            
            # Capture quote formatting
            current_path = await self.capture_element_screenshot(
                page, '.hidden-word-quote', f"{test_name}_current"
            )
            
            # Check baseline
            baseline_path = self.baselines_dir / f"{test_name}_baseline.png"
            
            if not baseline_path.exists():
                import shutil
                shutil.copy(str(current_path), str(baseline_path))
                logger.info(f"Created baseline for {test_name}")
                
                return VisualTestResult(
                    test_name=test_name,
                    passed=True,
                    similarity_score=100.0,
                    difference_percentage=0.0,
                    baseline_path=str(baseline_path),
                    current_path=str(current_path),
                    diff_path="",
                    threshold_met=True
                )
            
            return self.compare_images(baseline_path, current_path, test_name)
            
        except Exception as e:
            logger.error(f"Quote formatting test failed: {e}")
            return VisualTestResult(
                test_name=test_name,
                passed=False,
                similarity_score=0.0,
                difference_percentage=100.0,
                baseline_path="",
                current_path="",
                diff_path="",
                threshold_met=False,
                error_message=str(e)
            )
    
    async def test_responsive_design(self, page) -> Dict[str, VisualTestResult]:
        """Test responsive design across different viewports"""
        viewports = {
            "desktop": {"width": 1920, "height": 1080},
            "tablet": {"width": 768, "height": 1024},
            "mobile": {"width": 375, "height": 667}
        }
        
        results = {}
        
        for device_name, viewport in viewports.items():
            try:
                # Set viewport
                await page.set_viewport_size(viewport)
                await page.wait_for_timeout(1000)  # Let layout settle
                
                # Capture screenshot
                current_path = await self.capture_full_page_screenshot(
                    page, f"responsive_{device_name}_current"
                )
                
                # Check baseline
                baseline_path = self.baselines_dir / f"responsive_{device_name}_baseline.png"
                
                if not baseline_path.exists():
                    import shutil
                    shutil.copy(str(current_path), str(baseline_path))
                    logger.info(f"Created baseline for responsive_{device_name}")
                    
                    results[device_name] = VisualTestResult(
                        test_name=f"responsive_{device_name}",
                        passed=True,
                        similarity_score=100.0,
                        difference_percentage=0.0,
                        baseline_path=str(baseline_path),
                        current_path=str(current_path),
                        diff_path="",
                        threshold_met=True
                    )
                else:
                    results[device_name] = self.compare_images(
                        baseline_path, current_path, f"responsive_{device_name}"
                    )
                    
            except Exception as e:
                logger.error(f"Responsive test failed for {device_name}: {e}")
                results[device_name] = VisualTestResult(
                    test_name=f"responsive_{device_name}",
                    passed=False,
                    similarity_score=0.0,
                    difference_percentage=100.0,
                    baseline_path="",
                    current_path="",
                    diff_path="",
                    threshold_met=False,
                    error_message=str(e)
                )
        
        return results
    
    async def test_ui_state_changes(self, page) -> Dict[str, VisualTestResult]:
        """Test visual changes during different UI states"""
        states = {
            "initial_load": lambda: page.wait_for_selector('.manuscript-container'),
            "input_focused": lambda: page.locator('.search-input').click(),
            "button_hover": lambda: page.locator('.reveal-button').hover(),
            "voice_button_active": lambda: page.locator('.voice-button').click()
        }
        
        results = {}
        
        for state_name, state_action in states.items():
            try:
                # Execute state action
                await state_action()
                await page.wait_for_timeout(500)  # Let state settle
                
                # Capture state
                current_path = await self.capture_full_page_screenshot(
                    page, f"ui_state_{state_name}_current"
                )
                
                # Check baseline
                baseline_path = self.baselines_dir / f"ui_state_{state_name}_baseline.png"
                
                if not baseline_path.exists():
                    import shutil
                    shutil.copy(str(current_path), str(baseline_path))
                    logger.info(f"Created baseline for ui_state_{state_name}")
                    
                    results[state_name] = VisualTestResult(
                        test_name=f"ui_state_{state_name}",
                        passed=True,
                        similarity_score=100.0,
                        difference_percentage=0.0,
                        baseline_path=str(baseline_path),
                        current_path=str(current_path),
                        diff_path="",
                        threshold_met=True
                    )
                else:
                    results[state_name] = self.compare_images(
                        baseline_path, current_path, f"ui_state_{state_name}"
                    )
                    
            except Exception as e:
                logger.error(f"UI state test failed for {state_name}: {e}")
                results[state_name] = VisualTestResult(
                    test_name=f"ui_state_{state_name}",
                    passed=False,
                    similarity_score=0.0,
                    difference_percentage=100.0,
                    baseline_path="",
                    current_path="",
                    diff_path="",
                    threshold_met=False,
                    error_message=str(e)
                )
        
        return results
    
    def generate_visual_test_report(self, results: List[VisualTestResult]) -> Dict[str, Any]:
        """Generate comprehensive visual test report"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        avg_similarity = np.mean([r.similarity_score for r in results])
        avg_difference = np.mean([r.difference_percentage for r in results])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_visual_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_similarity": avg_similarity,
                "average_difference": avg_difference
            },
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "similarity_score": r.similarity_score,
                    "difference_percentage": r.difference_percentage,
                    "threshold_met": r.threshold_met,
                    "baseline_path": r.baseline_path,
                    "current_path": r.current_path,
                    "diff_path": r.diff_path,
                    "error_message": r.error_message
                }
                for r in results
            ],
            "recommendations": self._generate_visual_recommendations(results)
        }
        
        # Save report
        report_path = self.results_dir / f"visual_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Visual test report saved to: {report_path}")
        return report
    
    def _generate_visual_recommendations(self, results: List[VisualTestResult]) -> List[str]:
        """Generate recommendations based on visual test results"""
        recommendations = []
        
        failed_results = [r for r in results if not r.passed]
        
        if failed_results:
            high_diff_tests = [r for r in failed_results if r.difference_percentage > 10]
            if high_diff_tests:
                recommendations.append(
                    f"High visual differences detected in {len(high_diff_tests)} tests. "
                    "Review UI changes or update baselines if changes are intentional."
                )
            
            font_related = [r for r in failed_results if "persian" in r.test_name.lower()]
            if font_related:
                recommendations.append(
                    "Persian/Arabic text rendering issues detected. "
                    "Verify font loading and Unicode support."
                )
            
            responsive_issues = [r for r in failed_results if "responsive" in r.test_name.lower()]
            if responsive_issues:
                recommendations.append(
                    f"Responsive design issues found on {len(responsive_issues)} viewport(s). "
                    "Review CSS media queries and layout rules."
                )
        
        avg_similarity = np.mean([r.similarity_score for r in results])
        if avg_similarity < 95:
            recommendations.append(
                f"Overall visual similarity is {avg_similarity:.1f}%. "
                "Consider updating visual baselines or investigating systematic changes."
            )
        
        return recommendations