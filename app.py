import yaml
import utils
import streamlit as st
import lightgbm
from sklearn import *

mod_class_dict = {"Decision Tree Regressor": tree.DecisionTreeRegressor,
                  "Light Gradient Boosting Machine": lightgbm.LGBMRegressor,
                  "Linear Regression": linear_model.LinearRegression}

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)

# step1: checking the absence of CSV file
st.header("Demand Prediction")
st.write("This application allows to forecast the unit sales of Walmart retail goods ")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file")
if not uploaded_file:
    st.write("Plase, load CSV file to start")
    st.stop()

# step2: converting csv file into pandas DataFrame.
df = utils.to_df(uploaded_file)
st.subheader("Data Preview")
st.dataframe(df.head())

# step3: environment setup
reg = utils.create_setup(df)

# step4: setting parameters
st.sidebar.header("Modeling Preferences")
action = st.sidebar.selectbox("Choose an action", ["Custom Modeling", "Auto ML"])

# step 5: modeling
# automl path
if action == "Auto ML":
    st.subheader("Train")
    final = utils.autoML(df)
    predictions = utils.predict_model(final)
    st.subheader("Generate predictions")
    utils.plot(predictions)

# custom modeling path
elif action == "Custom Modeling":
    models_chosen = st.sidebar.multiselect("Choose Models to build", ["Decision Tree Regressor",
                                                                      "Light Gradient Boosting Machine",
                                                                      "Linear Regression"],
                                           default=["Decision Tree Regressor"])
    tune_button = st.sidebar.checkbox("Enable auto-tuning hyperparameters")
    final = utils.train(models_chosen, tune_button)
    if len(models_chosen) > 1:
        model_selected = st.sidebar.selectbox("Choose Models to make predictions", models_chosen)
        for i in final:
            if isinstance(i, (mod_class_dict[model_selected])):
                final=i
    predictions = utils.predict_model(final)
    st.subheader("Generate predictions")
    utils.plot(predictions)