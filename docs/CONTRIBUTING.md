# Contributing to PIPER - AI Classroom Analyzer

Thank you for your interest in contributing to PIPER! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Basic knowledge of computer vision and machine learning
- Familiarity with OpenCV, PyTorch, and TensorFlow

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/classanalyzer.git
   cd classanalyzer
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run Tests**
   ```bash
   python test_system.py
   pytest tests/
   ```

## 🎯 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include system information (OS, Python version, GPU details)
- Provide steps to reproduce the issue
- Include error messages and logs

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain how it fits with the project goals

### Code Contributions

#### 1. Choose an Issue
- Look for issues labeled "good first issue" or "help wanted"
- Comment on the issue to indicate you're working on it

#### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

#### 3. Make Changes
- Follow the coding standards (see below)
- Write tests for new functionality
- Update documentation as needed

#### 4. Test Your Changes
```bash
# Run all tests
python test_system.py
pytest tests/

# Check code formatting
black . && isort .

# Type checking
mypy main_app.py run_app.py
```

#### 5. Submit Pull Request
- Push your branch to your fork
- Create a pull request with a clear description
- Link to any related issues

## 📝 Coding Standards

### Python Style
- Follow PEP 8
- Use Black for code formatting: `black .`
- Use isort for import sorting: `isort .`
- Maximum line length: 88 characters

### Code Quality
- Write docstrings for all functions and classes
- Use type hints where possible
- Keep functions focused and small
- Use meaningful variable names

### Example Code Style
```python
def detect_faces(
    image: np.ndarray, 
    confidence_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Detect faces in an image using MediaPipe.
    
    Args:
        image: Input image as numpy array
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of face detection results with bounding boxes
    """
    # Implementation here
    pass
```

### Testing
- Write unit tests for new functions
- Use pytest for testing framework
- Aim for >80% test coverage
- Test edge cases and error conditions

### Documentation
- Update README.md for new features
- Add docstrings to all public functions
- Include usage examples
- Update configuration documentation

## 🏗️ Project Structure

```
classanalyzer/
├── main_app.py              # Main application entry point
├── run_app.py               # Full AI suite application
├── models/                  # AI model implementations
│   ├── face_detection.py
│   ├── face_recognition.py
│   └── ...
├── utils/                   # Utility functions
│   ├── enhanced_tracking_overlay.py
│   ├── comprehensive_analyzer.py
│   └── ...
├── scripts/                 # Additional scripts
├── config/                  # Configuration files
├── tests/                   # Test files
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Use descriptive test names

### Integration Tests
- Test component interactions
- Test with real camera input (when available)
- Test model loading and inference

### Performance Tests
- Benchmark critical functions
- Test memory usage
- Verify FPS requirements

## 📚 Documentation

### Code Documentation
- Use Google-style docstrings
- Document all parameters and return values
- Include usage examples

### User Documentation
- Update README.md for user-facing changes
- Add configuration examples
- Include troubleshooting information

## 🔄 Release Process

### Version Numbering
- Follow Semantic Versioning (SemVer)
- Format: MAJOR.MINOR.PATCH
- Update version in setup.py

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number bumped
- [ ] Changelog updated
- [ ] Performance benchmarks run

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional communication

### Communication
- Use GitHub issues for bug reports and feature requests
- Use pull request comments for code review
- Be patient with response times

## 🎯 Priority Areas

We especially welcome contributions in these areas:

1. **Performance Optimization**
   - GPU acceleration improvements
   - Memory usage optimization
   - Real-time processing enhancements

2. **Model Improvements**
   - Better face recognition accuracy
   - Enhanced engagement detection
   - Reduced false positives

3. **Platform Support**
   - macOS compatibility
   - Linux optimization
   - Mobile device support

4. **Documentation**
   - Tutorial videos
   - API documentation
   - Deployment guides

5. **Testing**
   - Automated testing
   - Performance benchmarks
   - Cross-platform testing

## 📞 Getting Help

- Check existing issues and documentation first
- Open a GitHub issue for bugs or questions
- Tag maintainers for urgent issues
- Be patient - this is an open source project

## 🙏 Recognition

Contributors will be:
- Listed in the README.md contributors section
- Mentioned in release notes
- Invited to join the core team (for significant contributions)

Thank you for contributing to PIPER - AI Classroom Analyzer!
