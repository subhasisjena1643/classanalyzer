# 🚀 Quick Installation Guide

## One-Click Installation

### Windows Users
```cmd
# Double-click this file for instant setup
start.bat
```

### macOS/Linux Users
```bash
# Run this command for instant setup
./start.sh
```

### Universal (All Platforms)
```bash
# Automated installation
python tools/install.py

# Interactive setup
python tools/quick_start.py

# Check compatibility first
python tools/check_requirements.py
```

## Manual Installation

### Step 1: Requirements
- **Python 3.8+** 
- **4GB+ RAM**
- **5GB free storage**
- **Camera/webcam**

### Step 2: Install
```bash
# Clone repository
git clone https://github.com/subhasisjena1643/classanalyzer.git
cd classanalyzer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run
```bash
# Main application (recommended)
python main_app.py

# Full AI suite
python run_app.py
```

## Troubleshooting

### Common Issues
- **Camera not detected**: Check permissions and connections
- **Import errors**: Run `python tools/install.py`
- **Performance issues**: System auto-optimizes based on hardware

### Need Help?
- **Detailed troubleshooting**: `docs/FAQ.md`
- **System diagnostics**: `python tools/check_requirements.py`
- **Complete documentation**: `README.md`
- **Quick test**: `python tests/test_system.py`

## Supported Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Ubuntu 18.04+
- ✅ Other Linux distributions

**🎯 Most users can just run `start.bat` (Windows) or `./start.sh` (macOS/Linux) for instant setup!**
