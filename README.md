# parkcam Project

## Overview
The parkcam project is designed to capture images from a camera and perform Automatic License Plate Recognition (ALPR) using the fast-alpr library. This application can be used for parking management systems, security monitoring, and other use cases where license plate identification is required.

## Features
- Capture images from a camera.
- Process images to detect and recognize license plates.
- Utility functions for image processing and configuration management.
- Unit tests to ensure core functionalities work as expected.

## Project Structure
```
parkcam
├── src
│   ├── parkcam
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── camera.py
│   │   ├── alpr.py
│   │   └── utils.py
│   └── scripts
│       └── fast-alpr.py
├── tests
│   └── test_basic.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github
│   └── workflows
│       └── ci.yml
└── README.md
```

## Installation
To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```

## Usage
To start the application, run the main script:
```
python src/parkcam/main.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.