# Pull Request Time-to-Merge Predictor

## Project Overview
This project is a machine learning prototype designed to predict the **Time to Merge** (in hours) for GitHub Pull Requests. By analyzing metadata available at or shortly after PR creation, the system provides teams with an estimated wait time, helping to optimize workflows and developer productivity.

The model is trained on data from two prominent open-source repositories:
* **microsoft/vscode**
* **excalidraw/excalidraw**

---

## Project Structure
├── data/                   # CSV storage (Raw and Processed)
├── src/
│   ├── get_data.py         # GitHub API Ingestion script
│   ├── preprocess.py       # Preprocessing
│   ├── train.py            # Training, and Evaluation
│   └── app.py              # Streamlit Live Demonstration
├── model.pkl           # Saved Random Forest Model
├── model_columns.pkl   # Saved feature list for alignment
├── requirements.txt    # Install library requirements
└── README.md           # Documentation

## Setup and Execution

### 1. GitHub Authentication
To fetch data from the GitHub API, you need a Personal Access Token (PAT). Set it as an environment variable:

**Linux/macOS:**
```bash
export GITHUB_TOKEN='your_token_here'
```

**Windows (PowerShell):**
```powershell
$env:GITHUB_TOKEN='your_token_here'
```

### 2. Execution Pipeline
Follow these steps in order to replicate the results:

1.  **Install Requirements**
    ```bash
    uv add -r requirements.txt
    ```

2.  **Data Ingestion**: Get PR data from GitHub.
    ```bash
    python src/get_data.py
    ```
    *Output*: `data/prs_merged_cleaned.csv`

3.  **Data Preprocessing**: Prepare the data for modeling.
    ```bash
    python src/preprocess.py
    ```
    *Output*: `data/prs_processed.csv`

4.  **Model Training**: Train the model and evaluate its performance.
    ```bash
    python src/train.py
    ```

5.  **Launch Demo**: Start the interactive Streamlit dashboard.
    ```bash
    streamlit run src/app.py
    ```


