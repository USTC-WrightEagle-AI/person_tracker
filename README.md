# Team Navigation Repo

This repository contains the person tracking stack for the Robocup@Home team.

## Main Structure

- `person_following`: Main detection files and configurations.



## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/USTC-WrightEagle-AI/person_tracker.git
   cd person_tracker
   ```

2. Install ROS:
   ```bash
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```

3. Build the workspace:
   ```bash
   catkin_make
   ```

4. Source the setup script:
   ```bash
   source devel/setup.bash
   ```

5. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
6.Install yolo
   ```bash
   pip install ultralytics

   ```


## Usage

To start the navigation stack:

```bash
python person_following/person_detection.py
```

```bash
python person_following/person_tracker.py
```