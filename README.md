# Low Power Event-RGB Based Spacecraft Pose Estimation 

Work in progress (February 2026)

## 1. Clone repository
TODO: Write


## 2. Install required packages
### 2.1. Create new virtual environment
python -m venv venv

### 2.2. Activate virtual environment
Windows -> .\venv\Scripts\Activate.ps1
Linux -> source venv/bin/activate

### 2.3. Install all packages from requirements.txt
pip install -r requirements.txt


## 3. Download FRESH dataset
FRESH dataset available in: https://zenodo.org/records/15861758

Follow these instructions for downloading and extracting into the project folders:

### 3.1. Go to your project folder
cd ~/Workspace/Low_Power_Satellite_6DoF_Pose_Estimation

### 3.2. Create the download and dataset folders (if not already there)
mkdir -p _downloads _dataset

### 3.3. Download all 5 files into _downloads (using wget or curl – wget is usually faster)
cd _downloads

wget -c "https://zenodo.org/records/15861758/files/models.zip?download=1"     -O models.zip
wget -c "https://zenodo.org/records/15861758/files/synthetic.zip?download=1"  -O synthetic.zip
wget -c "https://zenodo.org/records/15861758/files/real.zip.001?download=1"   -O real.zip.001
wget -c "https://zenodo.org/records/15861758/files/real.zip.002?download=1"   -O real.zip.002
wget -c "https://zenodo.org/records/15861758/files/real.zip.003?download=1"   -O real.zip.003

### 3.4. Go back one level and extract everything to _dataset
cd ..

#### Extract all zips (models, synthetic, and the real multi-part)
unzip _downloads/models.zip    -d _dataset/fresh
unzip _downloads/synthetic.zip -d _dataset/fresh

#### Extract multi-part real.zip (only need to unzip the .001 file – it auto-detects the others)
(sudo apt update && sudo apt install p7zip-full -y)
7z x _downloads/real.zip.001 -o_dataset/fresh

### 3.5. Optional: clean up the downloads folder after successful extraction
rm -rf _downloads


## 4. Download animals dataset
cd _downloads

wget -O animals.zip "https://storage.googleapis.com/roboflow-platform-regional-exports/pwYAXv9BTpqLyFfgQoPZ/9GCQoQCqZ8OOdLkCpKlJ/2/yolov8.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=481589474394-compute%40developer.gserviceaccount.com%2F20260309%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260309T211646Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&X-Goog-Signature=08a3843a0866661424f2237016734950e005e1c704961960a82a3b7ac2c20cf976ef9bc58b214bbabba289ef8f05c1822e949e90aa3044356ef4d49a1ecc9f8afee59acab63d5a5b4683ebce7f9b3859eb0d42bdb21aacb57dc0a3aa1ed2c9f236d75977bdef12781429ae16534d2c4ae1588ddfbb8f1225b1d126bc90a1a165f63e98c36d6c359077f54d5e44e1c5c4e49bc71a171e9a4feaa72c6c02ec85f5f14f91e7901e64659a985e8f7fdf7262e45cc95f83375ff47d1e2e1336115f955ce40a568407da27827e96a4f0840a19c6c4708dd8477251805ad6025d245aae3e580bf69059a95e23a3945f1bca7e2e4d9ff6c3bb208028599500878b7ea02e"

cd ..

unzip _downloads/animals.zip    -d _dataset/animals

## 4. Download ImageNet dataset

cd _downloads

mkdir imagenet

cd imagenet

wget -O imagenet.zip "https://www.kaggle.com/competitions/imagenet-object-localization-challenge/download-directory/BfknyVadERJfaqszLb7Z%2Fversions%2F4q0p1zUqWPJWgWXDucL0%2Fdirectories%2FILSVRC"

cd ..

unzip _downloads/imagenet.zip    -d _dataset/imagenet

