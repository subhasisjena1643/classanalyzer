#!/usr/bin/env python3
"""
GitHub Readiness Verification Script
Checks if the codebase is ready for GitHub upload
"""

import os
import re
import sys
from pathlib import Path

def print_header():
    """Print verification header"""
    print("=" * 70)
    print("🔍 GitHub Readiness Verification")
    print("=" * 70)
    print("Checking if the codebase is ready for public GitHub upload...")
    print()

def check_sensitive_patterns():
    """Check for sensitive information patterns"""
    print("🔒 Checking for sensitive information...")
    
    sensitive_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',  # Passwords
        r'secret\s*=\s*["\'][^"\']+["\']',    # Secrets
        r'token\s*=\s*["\'][^"\']+["\']',     # Tokens
        r'api_key\s*=\s*["\'][^"\']+["\']',   # API keys
        r'sk-[A-Za-z0-9]{32,}',              # OpenAI API keys
        r'ghp_[A-Za-z0-9]{36}',              # GitHub tokens
        r'xoxb-[A-Za-z0-9-]{50,}',           # Slack tokens
    ]
    
    python_files = list(Path('.').rglob('*.py'))
    issues_found = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in sensitive_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Filter out common false positives
                    filtered_matches = []
                    for match in matches:
                        if not any(fp in match.lower() for fp in ['example', 'test', 'dummy', 'placeholder', 'localhost', '127.0.0.1', 'class', 'def ', 'import']):
                            filtered_matches.append(match)
                    
                    if filtered_matches:
                        issues_found.append((file_path, pattern, filtered_matches))
        
        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}")
    
    if issues_found:
        print("❌ Potential sensitive information found:")
        for file_path, pattern, matches in issues_found:
            print(f"   {file_path}: {matches[:3]}...")  # Show first 3 matches
        return False
    else:
        print("✅ No sensitive information detected")
        return True

def check_local_paths():
    """Check for hardcoded local paths"""
    print("\n📁 Checking for hardcoded local paths...")
    
    path_patterns = [
        r'[C-Z]:\\[^"\']*',  # Windows absolute paths
        r'/Users/[^/\s"\']+',  # macOS user paths
        r'/home/[^/\s"\']+',   # Linux user paths
        r'\\\\[^"\']*',        # UNC paths
    ]
    
    python_files = list(Path('.').rglob('*.py'))
    issues_found = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in path_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Filter out comments, documentation, and regex patterns
                    filtered_matches = []
                    for match in matches:
                        if not any(fp in match.lower() for fp in ['example', 'comment', '#', 'doc', 'regex', 'pattern', '[^', 'r\'', 'r"']):
                            filtered_matches.append(match)
                    
                    if filtered_matches:
                        issues_found.append((file_path, filtered_matches))
        
        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}")
    
    if issues_found:
        print("❌ Hardcoded local paths found:")
        for file_path, matches in issues_found:
            print(f"   {file_path}: {matches[:2]}...")  # Show first 2 matches
        return False
    else:
        print("✅ No hardcoded local paths detected")
        return True

def check_required_files():
    """Check if all required files exist"""
    print("\n📋 Checking required files...")
    
    required_files = [
        'README.md',
        'INSTALL.md',
        'LICENSE',
        'requirements.txt',
        'setup.py',
        '.gitignore',
        'main_app.py',
        'run_app.py',
        'start.bat',
        'start.sh',
        'config/app_config.yaml',
        'tools/install.py',
        'tests/test_system.py',
        'docs/CONTRIBUTING.md',
        'docs/FAQ.md',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_file_permissions():
    """Check file permissions and executability"""
    print("\n🔐 Checking file permissions...")
    
    executable_files = [
        'install.py',
        'quick_start.py',
        'test_system.py',
        'verify_github_ready.py',
    ]
    
    issues = []
    for file_path in executable_files:
        if os.path.exists(file_path):
            # Check if file starts with shebang
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#!/usr/bin/env python'):
                        print(f"✅ {file_path} - Proper shebang")
                    else:
                        issues.append(f"{file_path} - Missing or incorrect shebang")
            except Exception as e:
                issues.append(f"{file_path} - Could not read: {e}")
    
    if issues:
        print("⚠️  File permission issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ File permissions OK")
        return True

def check_documentation_quality():
    """Check documentation quality"""
    print("\n📖 Checking documentation quality...")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = [
            'installation',
            'usage',
            'features',
            'requirements',
            'license',
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️  README.md missing sections: {missing_sections}")
        else:
            print("✅ README.md has all required sections")
        
        # Check if README is comprehensive (length check)
        if len(readme_content) < 5000:
            print("⚠️  README.md might be too brief for a comprehensive project")
        else:
            print("✅ README.md is comprehensive")
        
        return len(missing_sections) == 0
        
    except Exception as e:
        print(f"❌ Could not check README.md: {e}")
        return False

def check_dependencies():
    """Check if dependencies are properly specified"""
    print("\n📦 Checking dependencies...")
    
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = f.read()
        
        # Check for version pinning
        lines = [line.strip() for line in requirements.split('\n') if line.strip() and not line.startswith('#')]
        unpinned = [line for line in lines if '==' not in line and '>=' not in line and line]
        
        if unpinned:
            print(f"⚠️  Unpinned dependencies: {unpinned[:5]}...")
        else:
            print("✅ All dependencies are properly pinned")
        
        # Check for essential packages
        essential_packages = ['opencv-python', 'mediapipe', 'torch', 'tensorflow', 'numpy']
        missing_essential = []
        
        for package in essential_packages:
            if package not in requirements:
                missing_essential.append(package)
        
        if missing_essential:
            print(f"❌ Missing essential packages: {missing_essential}")
            return False
        else:
            print("✅ All essential packages included")
            return True
            
    except Exception as e:
        print(f"❌ Could not check requirements.txt: {e}")
        return False

def generate_summary_report():
    """Generate a summary report"""
    print("\n" + "=" * 70)
    print("📊 GITHUB READINESS SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Sensitive Information", check_sensitive_patterns),
        ("Local Paths", check_local_paths),
        ("Required Files", check_required_files),
        ("File Permissions", check_file_permissions),
        ("Documentation", check_documentation_quality),
        ("Dependencies", check_dependencies),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    print("\n📋 Final Results:")
    passed = 0
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name:<20} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n🎯 Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 CODEBASE IS GITHUB READY!")
        print("✅ You can safely upload this to GitHub")
        print("\n🚀 Next steps:")
        print("1. Initialize git repository: git init")
        print("2. Add files: git add .")
        print("3. Commit: git commit -m 'Initial commit'")
        print("4. Add remote: git remote add origin <your-repo-url>")
        print("5. Push: git push -u origin main")
        return True
    else:
        print(f"\n⚠️  {total - passed} issue(s) need to be fixed before GitHub upload")
        print("Please address the failed checks above")
        return False

def main():
    """Main verification function"""
    print_header()
    success = generate_summary_report()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
