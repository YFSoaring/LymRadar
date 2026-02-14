1. Project Overview
  LymRadar is a deep learning framework designed to predict Primary Treatment Failure (PTF) in DLBCL patients. 
2. Environment Setup
  To ensure the code runs correctly, please set up a specific Python environment.
  2.1 Create Virtual Environment
    It is recommended to use Conda to manage the environment and avoid dependency conflicts.
  2.2 Install Dependencies
    Install all required Python packages listed in requirements.txt.
3. Data Preparation
  Before running the model, raw medical images must be preprocessed into a unified format and size using the DataPre.py script.
  3.1 Input Requirements
    Data Type: Co-registered PET and CT images.
    File Formats: .nrrd, .nii, or .nii.gz.
  3.2 Preprocessing Workflow (DataPre.py)
    Running the preprocessing script performs the following steps:
    Loading: Reads the paired PET/CT volumes.
    Resizing: Resamples and crops/pads images to a fixed dimension of 256 × 128 × 128 (Depth × Height × Width).
    Saves the final tensors as .npy files (NumPy array) for model input.
4. Inference and Testing
  Once the data is converted to .npy format, use Test.py to load the model weights and generate predictions.
5. File Structure Structure
  Please maintain the following directory structure for seamless execution:
Project_Root/
  ├── DataPre.py (Script for data preprocessing)
  ├── Test.py (Script for model inference)
  ├── LymRadar.pt (Pre-trained model weights)
  ├── requirements.txt (List of dependencies)
  └── README.md (Documentation)