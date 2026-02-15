import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic Survival Prediction")

st.title("Titanic Survival Prediction App")

# =====================================================
# MODEL SELECTION
# =====================================================
model_name = st.selectbox(
    "Select Model",
    [
        "logistic",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost"
    ]
)

# =====================================================
# LOAD DATA (FOR EVALUATION)
# =====================================================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("train.csv")

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

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Title'] = df['Title'].map(
        {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    )

    features = [
        'Pclass','Sex','Age','SibSp','Parch','Fare',
        'Embarked','FamilySize','IsAlone','Title'
    ]

    X = df[features]
    y = df['Survived']

    return X, y, features


X, y, features = load_and_prepare_data()

# =====================================================
# LOAD MODEL & SCALER (ON DEMAND)
# =====================================================
@st.cache_resource
def load_model(model_name):
    model = joblib.load(f"model/saved_models/{model_name}.pkl")
    scaler = joblib.load("model/saved_models/scaler.pkl")
    return model, scaler


# =====================================================
# EVALUATION SECTION
# =====================================================
st.header("Model Evaluation")

if st.button("Evaluate Selected Model"):
    with st.spinner("Loading model and evaluating..."):
        model, scaler = load_model(model_name)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)

        st.metric("Accuracy", f"{accuracy:.2f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

# =====================================================
# PREDICTION SECTION
# =====================================================
st.header("Predict on New Data")

uploaded_file = st.file_uploader(
    "Upload test data of Titanic passengers for survival prediction:",
    type=["csv"]
)

if uploaded_file is not None:
    if st.button("Run Prediction"):
        try:
            model, scaler = load_model(model_name)

            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview")
            st.write(df.head())

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

            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
            df['Title'] = df['Title'].map(
                {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
            )

            X_new = df[features]
            X_new_scaled = scaler.transform(X_new)

            predictions = model.predict(X_new_scaled)
            probabilities = model.predict_proba(X_new_scaled)[:, 1]

            df['Survived_Prediction'] = predictions
            df['Survival_Probability'] = probabilities

            st.subheader("Prediction Results")
            st.write(df[['Survived_Prediction', 'Survival_Probability']])

            st.success("Prediction completed successfully")

        except Exception as e:
            st.error("Error during prediction")
            st.exception(e)
