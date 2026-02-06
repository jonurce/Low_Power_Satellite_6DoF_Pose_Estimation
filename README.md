# Low Power Event-RGB Based Spacecraft Pose Estimation 

Work in progress (February 2026)

## 1. Clone repository
TODO: Write


## 2. Download FRESH dataset
FRESH dataset available in: https://zenodo.org/records/15861758

Follow these instructions for downloading and extracting into the project folders:

### 2.1. Go to your project folder
cd ~/Workspace/Low_Power_Satellite_6DoF_Pose_Estimation

### 2.2. Create the download and dataset folders (if not already there)
mkdir -p _downloads _dataset

### 2.3. Download all 5 files into _downloads (using wget or curl – wget is usually faster)
cd _downloads

wget -c "https://zenodo.org/records/15861758/files/models.zip?download=1"     -O models.zip
wget -c "https://zenodo.org/records/15861758/files/synthetic.zip?download=1"  -O synthetic.zip
wget -c "https://zenodo.org/records/15861758/files/real.zip.001?download=1"   -O real.zip.001
wget -c "https://zenodo.org/records/15861758/files/real.zip.002?download=1"   -O real.zip.002
wget -c "https://zenodo.org/records/15861758/files/real.zip.003?download=1"   -O real.zip.003

### 2.4. Go back one level and extract everything to _dataset
cd ..

#### Extract all zips (models, synthetic, and the real multi-part)
unzip _downloads/models.zip    -d _dataset/
unzip _downloads/synthetic.zip -d _dataset/

#### Extract multi-part real.zip (only need to unzip the .001 file – it auto-detects the others)
(sudo apt update && sudo apt install p7zip-full -y)
7z x _downloads/real.zip.001 -o_dataset/

### 2.5. Optional: clean up the downloads folder after successful extraction
rm -rf _downloads

## 3. Install required packages
### 3.1. Create new virtual environment
python -m venv venv

### 3.2. Activate virtual environment
Windows -> .\venv\Scripts\Activate.ps1
Linux -> source venv/bin/activate

### 3.3. Install all packages from requirements.txt
pip install -r requirements.txt