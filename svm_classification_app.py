import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("SVM Happy Prediction App")
st.write("Use the `happydata.csv` dataset to predict whether a person is happy based on survey attributes.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("happydata.csv")
    if "�vents" in df.columns:
        df = df.rename(columns={"�vents": "events"})
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Feature Description")
st.write(
    "`infoavail`, `housecost`, `schoolquality`, `policetrust`, `streetquality`, `events` are input features. "
    "`happy` is the target label (0 or 1)."
)

# Prepare features and target
X = df.drop("happy", axis=1)
y = df["happy"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
)

st.sidebar.header("Model Hyperparameters")
kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
C = st.sidebar.slider("Regularization C", 0.01, 10.0, 0.1, 0.01)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)

# Create and train model
model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Evaluation")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

with st.expander("Show classification report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

with st.expander("Show confusion matrix"):
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

st.subheader("Make a Prediction")
infoavail = st.number_input("Info Available", min_value=0, max_value=10, value=3)
housecost = st.number_input("House Cost", min_value=0, max_value=10, value=3)
schoolquality = st.number_input("School Quality", min_value=0, max_value=10, value=3)
policetrust = st.number_input("Police Trust", min_value=0, max_value=10, value=3)
streetquality = st.number_input("Street Quality", min_value=0, max_value=10, value=2)
events = st.number_input("Events", min_value=0, max_value=10, value=4)

if st.button("Predict Happiness"):
    input_data = [[infoavail, housecost, schoolquality, policetrust, streetquality, events]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    labels = {0: "Not Happy", 1: "Happy"}
    st.success(f"Predicted outcome: {labels[int(prediction[0])]}.")
