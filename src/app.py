import streamlit as st
import joblib
import pandas as pd
import os

# 1. SETTINGS & MODEL LOADING
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "model_columns.pkl"

st.set_page_config(page_title="PR Merge Predictor", layout="centered")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

model, model_columns = load_assets()

# 2. UI HEADER
st.title("Pull Request Merge Predictor")
st.markdown("""
This tool predicts the **Time to Merge** (in hours) for GitHub Pull Requests. 
It uses a Random Forest model trained on data from `vscode` and `excalidraw`.
""")

if model is None:
    st.error(" Model files not found! Please ensure 'model.pkl' and 'model_columns.pkl' are in the same directory.")
    st.stop()

# 3. USER INPUTS
st.subheader(" PR Characteristics")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    repo = st.selectbox(
        "Target Repository", 
        ["microsoft/vscode", "excalidraw/excalidraw"]
    )
    
    author_assoc = st.selectbox(
        "Author Association", 
        ["MEMBER", "CONTRIBUTOR", "NONE", "COLLABORATOR", "OWNER"]
    )

    # Find which associations were actually present during training
    trained_associations = [
        col.replace("author_assoc_", "") 
        for col in model_columns 
        if col.startswith("author_assoc_")
    ]
    
    if author_assoc not in trained_associations:
        st.info(f" Note: '{author_assoc}' was not seen during training. The model will treat this as a generic baseline.")

    is_draft = st.checkbox("Is this a Draft PR?")

with col2:
    additions = st.number_input("Lines Added (+)", min_value=0, value=50, step=10)
    deletions = st.number_input("Lines Deleted (-)", min_value=0, value=10, step=10)
    changed_files = st.number_input("Files Changed", min_value=1, value=2)
    num_commits = st.number_input("Number of Commits", min_value=1, value=1)

# Advanced metadata tucked in an expander
with st.expander("Additional Metadata (Optional)"):
    title_len = st.slider("Title Length (characters)", 0, 200, 50)
    body_len = st.slider("Description Length (characters)", 0, 5000, 500)
    num_labels = st.number_input("Number of Labels applied", 0, 20, 1)

# 4. PREDICTION LOGIC ---
if st.button("Calculate Predicted Merge Time", type="primary"):
    # Create input dictionary
    input_dict = {
        "additions": additions,
        "deletions": deletions,
        "changed_files": changed_files,
        "num_commits": num_commits,
        "is_draft": int(is_draft),
        "title_len": title_len,
        "body_len": body_len,
        "num_labels": num_labels,
        "author_assoc": author_assoc,
        "repo": repo
    }

    # 1. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 2. One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # 3. Align Columns
    # Create a DataFrame with the exact same columns
    final_features = pd.DataFrame(columns=model_columns)
    # Merge the user's data into that shape, filling missing categories with 0
    final_features = pd.concat([final_features, input_encoded]).fillna(0)
    # Ensure the order is exactly the same
    final_features = final_features[model_columns]

    # 4. Predict
    prediction = model.predict(final_features)[0]

    # 5. RESULTS DISPLAY
    st.divider()
    st.subheader("Prediction Result")
    
    # Visualizing the output
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(label="Estimated Wait", value=f"{round(prediction, 1)} Hours")
    
    with col_res2:
        if prediction < 24:
            st.success(" **Fast Track!!!** This PR is likely to be merged within a single day.")
        elif prediction < 72:
            st.warning(" **Standard Review!** Expect a merge within 2-3 business days.")
        else:
            st.error(" **Complex Change!** This PR may require significant review time (3+ days).")

st.markdown("---")
st.caption("Case Study Assignment: Predicting Pull Request Time to Merge | Built with Streamlit & Scikit-Learn")