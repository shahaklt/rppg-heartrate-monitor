# Getting UBFC-rPPG (rPPG validation set)

The official source is a Google Drive folder:
https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk

Drive rate-limits the large `.avi` files ("Too many users have downloaded this
file recently"), so `gdown` often only retrieves the small helper files. Two
reliable fallbacks:

## Option A — Kaggle mirror (recommended)
1. Make a free Kaggle account, then Account → "Create New API Token" → downloads
   `kaggle.json`.
2. Put it at `C:\Users\<you>\.kaggle\kaggle.json`.
3. From the project venv:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install kaggle
   .\.venv\Scripts\kaggle.exe datasets download -d malekdinarito/ubfc-rppg-dataset -p data/raw/ubfc_rppg --unzip
   ```
   (alt mirror: `ashfakyeafi/ubfc-2`)

## Option B — retry gdown later
Drive quota resets ~24 h:
```powershell
.\.venv\Scripts\python.exe -m gdown --folder "https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk" -O data/raw/ubfc_rppg
```

After download, each subject folder should contain `vid.avi` + `ground_truth.txt`.
Then validate rPPG:
```powershell
.\.venv\Scripts\python.exe -m verification.verify_rppg
```

UBFC-rPPG is only needed to quantify rPPG accuracy (Pearson r vs ground-truth
pulse, HR error). Emotion training uses WESAD and does not depend on it.
