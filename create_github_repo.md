# GitHub Repository Setup

Follow these steps to create and push this repository to GitHub:

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `video-detection-system`
   - **Description**: `PPE Detection and Intrusion Detection for MJPEG Video Streams from Raspberry Pi`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Initialize and Push Local Repository

Run these commands in your project directory:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PPE and Intrusion Detection System

- PPE detection with YOLO model
- Intrusion detection with zone setup
- MJPEG stream support for Raspberry Pi
- Optimized for low latency
- Complete documentation and setup instructions"

# Add your GitHub repository as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/video-detection-system.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. Go to your GitHub repository page
2. Verify all files are uploaded
3. Check that the README.md displays correctly

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
# Create repository and push in one command
gh repo create video-detection-system --public --source=. --remote=origin --push

# Or for private repository
gh repo create video-detection-system --private --source=. --remote=origin --push
```

## Repository Structure

Your repository will contain:
```
video-detection-system/
├── README.md                    # Main documentation
├── SETUP.md                     # Setup instructions
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── inference_pc_stream.py       # PPE detection script
├── intrusion_pc_stream.py       # Intrusion detection script
├── test_stream.py               # Stream connection test
├── create_github_repo.md        # This file
├── ppe-detection-main/          # PPE detection module
└── intrusion-main/              # Intrusion detection module
```

## Important Notes

- Model files (*.pt) are excluded via .gitignore due to size
- Users will need to download models separately (instructions in SETUP.md)
- Log files and temporary files are also excluded
- The repository is ready for collaboration and contributions