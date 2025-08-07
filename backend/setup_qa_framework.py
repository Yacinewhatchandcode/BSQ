#!/usr/bin/env python3
"""
Setup script for the Enhanced Playwright QA Framework
Installs dependencies, initializes Playwright, and prepares the testing environment
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description=""):
    """Run a command and handle errors"""
    logger.info(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    logger.info("✅ Python version is compatible")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    logger.info("📦 Installing Python dependencies...")
    
    requirements_file = Path("playwright_requirements.txt")
    if not requirements_file.exists():
        logger.error("playwright_requirements.txt not found")
        return False
    
    # Install pip requirements
    if not run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Installing Python packages"):
        return False
    
    logger.info("✅ Python dependencies installed")
    return True

def install_playwright_browsers():
    """Install Playwright browsers"""
    logger.info("🌐 Installing Playwright browsers...")
    
    # Install Playwright browsers
    if not run_command(f"{sys.executable} -m playwright install", "Installing Playwright browsers"):
        return False
    
    # Install system dependencies
    system = platform.system().lower()
    if system == "linux":
        if not run_command(f"{sys.executable} -m playwright install-deps", "Installing system dependencies"):
            logger.warning("Failed to install system dependencies automatically")
    
    logger.info("✅ Playwright browsers installed")
    return True

def create_directory_structure():
    """Create necessary directory structure"""
    logger.info("📁 Creating directory structure...")
    
    directories = [
        "qa_results",
        "qa_results/screenshots",
        "qa_results/baselines",
        "qa_results/diffs",
        "qa_results/bugs",
        "qa_results/reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {directory}")
    
    logger.info("✅ Directory structure created")
    return True

def validate_server_availability():
    """Check if the Baha'i interface server is running"""
    logger.info("🌐 Checking server availability...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Baha'i interface server is running")
            return True
        else:
            logger.warning(f"Server responded with status code: {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Server not accessible: {e}")
        logger.info("⚠️  Please start the server with: python main.py")
        return False

def create_sample_test_config():
    """Create a sample test configuration if it doesn't exist"""
    config_file = Path("playwright_config.yaml")
    
    if config_file.exists():
        logger.info("✅ Configuration file already exists")
        return True
    
    logger.info("📝 Creating sample configuration...")
    
    sample_config = """# Playwright QA Framework Configuration

# Test Execution Settings
execution:
  headless: false  # Set to true for CI/CD
  browser: chromium
  timeout: 30000
  viewport:
    width: 1920
    height: 1080

# Application Under Test
application:
  base_url: "http://localhost:8000"
  websocket_url: "ws://localhost:8000/ws"

# Visual Testing
visual:
  enable_screenshots: true
  screenshot_dir: "qa_results/screenshots"
  
# Performance Testing
performance:
  max_page_load_time: 5000
  max_response_time: 15000

# Reporting
reporting:
  enable_html_report: true
  enable_json_report: true
  report_directory: "qa_results/reports"

# Bug Detection
bug_detection:
  enable_console_monitoring: true
  enable_network_monitoring: true
  enable_error_tracking: true
"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(sample_config)
    
    logger.info("✅ Sample configuration created")
    return True

def run_basic_test():
    """Run a basic test to verify the framework is working"""
    logger.info("🧪 Running basic framework test...")
    
    test_script = """
import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from playwright_qa_framework import BahaiQAFramework

async def basic_test():
    framework = BahaiQAFramework()
    
    if await framework.setup_browser(headless=True):
        page_object = framework.page_object
        if await page_object.navigate():
            if await page_object.wait_for_page_load():
                print("✅ Basic test passed - Framework is working!")
                await framework.teardown_browser()
                return True
    
    print("❌ Basic test failed")
    await framework.teardown_browser()
    return False

if __name__ == "__main__":
    result = asyncio.run(basic_test())
    sys.exit(0 if result else 1)
"""
    
    # Write temporary test file
    test_file = Path("temp_basic_test.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    try:
        # Run the basic test
        success = run_command(f"{sys.executable} temp_basic_test.py", "Running basic test")
        
        # Clean up
        test_file.unlink()
        
        if success:
            logger.info("✅ Basic test passed!")
            return True
        else:
            logger.error("❌ Basic test failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run basic test: {e}")
        if test_file.exists():
            test_file.unlink()
        return False

def main():
    """Main setup function"""
    print("🚀 Enhanced Playwright QA Framework Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install Python dependencies
    if success and not install_python_dependencies():
        success = False
    
    # Install Playwright browsers
    if success and not install_playwright_browsers():
        success = False
    
    # Create directory structure
    if success and not create_directory_structure():
        success = False
    
    # Create sample configuration
    if success and not create_sample_test_config():
        success = False
    
    # Validate server (optional)
    server_running = validate_server_availability()
    
    # Run basic test
    if success and server_running and not run_basic_test():
        logger.warning("Basic test failed, but setup may still be valid")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Ensure the Baha'i interface server is running:")
        print("   python main.py")
        print("\n2. Run the QA framework:")
        print("   python enhanced_qa_framework.py")
        print("\n3. For continuous testing:")
        print("   python enhanced_qa_framework.py (select option 2)")
        print("\n4. View results in the qa_results/ directory")
        
        if not server_running:
            print("\n⚠️  Note: Start the server before running tests")
    else:
        print("❌ Setup failed. Please check the errors above.")
        print("\n🔧 Manual troubleshooting:")
        print("1. Ensure Python 3.8+ is installed")
        print("2. Install dependencies: pip install -r playwright_requirements.txt")
        print("3. Install Playwright: python -m playwright install")
        print("4. Check that the server is running at http://localhost:8000")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)