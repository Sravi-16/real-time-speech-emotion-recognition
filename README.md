# Real-Time Speech Emotion Recognition (Microphone-Based)

Complete end-to-end project with real microphone inference and Streamlit dashboard.

## What this project does

- Uses real `.wav` files from your local dataset folders (no fake predictions)
- Normalizes labels to: `happy`, `angry`, `neutral`, `sad`, `frustrated`, `confused`
- Splits by speaker to avoid leakage (unseen speaker test set)
- Trains a CNN + BiLSTM model in PyTorch
- Evaluates only on unseen test split
- Runs a real-time dashboard with Start/Stop microphone and live charts

## Final folder structure (current project)

```text
emotion project/
├── app.py
├── requirements.txt
├── README.md
├── backend/
├── frontend/
├── training/
├── dataset/
│   ├── RAVDESS/   (or ravdess, both supported)
│   ├── tess/      (optional)
│   └── cremad/    (optional)
└── models/
```

`RAVDESS` uppercase is supported and already mapped correctly in code.

## A) LOCAL RUN (Windows PowerShell) - where to run

Run all commands below **inside**:
`C:\Users\thriv\Desktop\emotion project`

### 1) Create environment + install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train and evaluate (unseen test only)

```powershell
python -m training.train --dataset_root dataset --epochs 24 --batch_size 32 --learning_rate 0.001 --patience 5
```

### 3) Run dashboard (localhost)

```powershell
streamlit run app.py
```

Then open the URL shown in terminal (usually `http://localhost:8501`).

## B) GOOGLE COLAB - where to run

Open `training/ser_colab_notebook.ipynb` in Colab.

### Colab steps

1. Run install cell.
2. Ensure folders exist at:
   - `/content/dataset/ravdess`
   - `/content/dataset/tess`
   - `/content/dataset/cremad`
3. Mount Drive and copy your folders into `/content/dataset`.
4. Put this project code in `/content/project`.
5. Run training cell:
   - `!python -m training.train --dataset_root /content/dataset --epochs 24 --batch_size 32 --learning_rate 0.001 --patience 5`
6. Download/copy `models/emotion_cnn_lstm.pt` and `training/results/` back to local machine.
7. Run dashboard locally with `streamlit run app.py`.

## Output files after training

- `models/emotion_cnn_lstm.pt`
- `training/results/metadata_clean.csv`
- `training/results/file_checks.csv` (corrupted/skipped logs if any)
- `training/results/train_split.csv`
- `training/results/val_split.csv`
- `training/results/test_split.csv`
- `training/results/train_history.csv`
- `training/results/test_metrics.json`
- `training/results/confusion_matrix.png`

## Notes

- Test metrics are computed only on unseen test speakers.
- If microphone is busy, close other apps using mic and restart Streamlit.
