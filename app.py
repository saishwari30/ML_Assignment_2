import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Prediction")

st.title("Titanic Survival Prediction App")

# =============================
# MODEL SELECTION
# =============================
model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn",
     "naive_bayes", "random_forest", "xgboost"]
)

# =============================
# LOAD MODEL & SCALER (CACHED)
# =============================
@st.cache_resource
def load_artifacts(model_name):
    model = joblib.load(f"model/saved_models/{model_name}.pkl")
    scaler = joblib.load("model/saved_models/scaler.pkl")
    return model, scaler

# model, scaler = load_artifacts(model_name)
# @st.cache_resource
# def load_artifacts():
#     model = joblib.load("model/saved_models/logistic.pkl")
#     scaler = joblib.load("model/saved_models/scaler.pkl")
#     return model, scaler
#
# model, scaler = load_artifacts()


# =============================
# FILE UPLOAD
# =============================
st.subheader("Upload Titanic CSV for Prediction")

uploaded_file = st.file_uploader(
    "Upload Titanic CSV (without 'Survived' column)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview")
        st.write(df.head())

        # ---- Preprocessing (NO fitting) ----
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(
            ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
            'Rare'
        )
        df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # ---- Manual encoding (matches training) ----
        sex_map = {'male': 0, 'female': 1}
        embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}

        df['Sex'] = df['Sex'].map(sex_map)
        df['Embarked'] = df['Embarked'].map(embarked_map)
        df['Title'] = df['Title'].map(title_map)

        features = [
            'Pclass','Sex','Age','SibSp','Parch','Fare',
            'Embarked','FamilySize','IsAlone','Title'
        ]

        X = df[features]
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        df['Survived_Prediction'] = predictions
        df['Survival_Probability'] = probabilities

        st.subheader("Prediction Results")
        st.write(df[['Survived_Prediction', 'Survival_Probability']])

        st.success("Prediction completed successfully")

    except Exception as e:
        st.error("Error during prediction")
        st.exception(e)
