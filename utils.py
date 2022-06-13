import yaml
import pandas as pd
import streamlit as st
from pycaret.regression import *
import altair as alt
from sklearn.model_selection import train_test_split

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)


@st.cache(suppress_st_warning=True)
def to_df(file):
    """Convert input file into pandas dataframe.
    
    Keyword argument:
    file -- gzip file uploaded by user
    """

    df = pd.read_csv(file, compression="gzip", dtype=config["data_types"])
    df.drop("date", axis=1, inplace=True)

    return df

@st.cache(suppress_st_warning=True)
def create_setup(df):
    """Initiate pycaret regression environment.

    Keyword argument:
    df -- pandas dataframe of input data
    """

    reg01 = setup(data=df, target='demand', session_id=123, normalize=True, fold=2, numeric_features=config["numeric"],
            ordinal_features={"item_id": list(df["item_id"].unique())}, data_split_shuffle=False,
            fold_strategy="timeseries", silent=True)

def compare(models_chosen, show_metrics=True):
    """Build and return trained model objects chosen by user.

    Keyword arguments:
    models_chosen -- a list of the names of the algorithms chosen by user
    show_metrics -- whether to show metrics of models campared (default True)
    """

    models_to_create = [config["models_dict"][model] for model in models_chosen]

    if show_metrics is False:
        with st.spinner("Building..."):
            compared = compare_models(include=models_to_create, verbose=False, sort="MAE", n_select=len(models_to_create))
            return compared

    elif show_metrics is True:
        st.subheader("Train")
        st.write("Baseline metrics")
        with st.spinner("Building..."):
            compared = compare_models(include=models_to_create, verbose=False, sort="MAE", n_select=len(models_to_create))
            met = pull()
            met.drop(["R2", "MSE", "RMSE", "TT (Sec)"], axis=1, inplace=True)
            met.reset_index(drop=True, inplace=True)
            met.set_index(keys="Model", drop=True, inplace=True)
            met1 = met.style.set_properties(**{'background-color': "#4DBFD9"}, subset=['MAE'])

            st.dataframe(met1)

            return compared

def tune(compared, show_metric=False):
    """Tune and return tuned model objects chosen by user.
    
    Keyword arguments:
    compared -- a single or list of trained model objects
    show_metrics -- whether to show metrics of models campared (default False)
    """ 

    with st.spinner("Tuning..."):
        if isinstance(compared, list):
            allTunedMetrics_df = pd.DataFrame(columns=["MAE", "RMSLE", "MAPE"])
            tuned_all = []

            for i in range(len(compared)):
                tuned_compared = tune_model(estimator=compared[i], n_iter=2, optimize="MAE", verbose=show_metric)
                tuned_all.append(tuned_compared)
                met = pull()

                # getting name of final estimator to display later
                est_name = str(tuned_compared).split("(")[0]
                met.drop(["R2", "MSE", "RMSE"], axis=1, inplace=True)
                met.reset_index(inplace=True, drop=True)
                met = met.iloc[-2]
                met = pd.DataFrame(met.values, columns=[est_name], index=met.index)
                met = met.transpose()

                # concat metrics to general df
                allTunedMetrics_df = pd.concat([allTunedMetrics_df, met])

            allTunedMetrics_df = allTunedMetrics_df.sort_values("MAE", ascending=True)
            allTunedMetrics_df = allTunedMetrics_df.style.set_properties(**{'background-color': "#4DBFD9"}, subset=['MAE'])

            st.subheader("Train")
            st.write("Metrics after tuning")
            st.dataframe(allTunedMetrics_df)

            return tuned_all

        else:
            tuned_compared = tune_model(estimator=compared, n_iter=2, optimize="MAE", verbose=show_metric)
            met = pull()

            # getting name of final estimator to display later
            est_name = str(tuned_compared).split("(")[0]
            met.drop(["R2", "MSE", "RMSE"], axis=1, inplace=True)
            met.reset_index(inplace=True, drop=True)
            met = met.iloc[-2]
            met = pd.DataFrame(met.values, columns=[est_name], index=met.index)
            met = met.transpose()
            met = met.style.set_properties(**{'background-color': "#4DBFD9"}, subset=['MAE'])

            st.subheader("Train")
            st.write("Metrics after tuning")
            st.dataframe(met)

            return tuned_compared


def plot(predict_df):
    """Plot the predictions for specific items.

    Keyword arguments:
    predict_df -- dataframe with model predictions
    """

    df = pd.read_csv(config["filepath"], compression="gzip", dtype=config["data_types"])

    train, new = train_test_split(df, test_size=0.3, train_size=0.7, shuffle=False)
    train = train \
        .loc[(train['store_id'] == 'CA_3') & (train['item_id'].isin(['HOUSEHOLD_1_383', 'HOUSEHOLD_1_366', 'HOUSEHOLD_1_349'])),
            ['date', 'item_id','demand']]
    predict_df.drop(predict_df.tail(1).index,inplace=True)
    new["demand"] = predict_df["Label"].values
    new = new \
        .loc[(new['store_id'] == 'CA_3') & (new['item_id'].isin(['HOUSEHOLD_1_383', 'HOUSEHOLD_1_366', 'HOUSEHOLD_1_349'])), ['date', 'item_id','demand']]
    train["Type"] = "historical"
    new["Type"] = "predicted"
    res = pd.concat([train,new])

    chart = alt.Chart(res).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('demand:Q', title='Demand'),
        color=alt.Color('item_id', title='Item ID'),
        strokeDash='Type:N'
    ).properties(
        width=700
    )

    st.altair_chart(chart)

def train(model_chosen, tune_button):
    """Return baseline or tuned model object.

    model_chosen -- names of algorithms chosen by user
    tune_button --  whether to enable tuning of hyperparameters
    """

    if not tune_button:
        final = compare(models_chosen=model_chosen, show_metrics=True)
    else:
        compared = compare(models_chosen=model_chosen, show_metrics=False)
        final = tune(compared, show_metric=True)
    return final

def autoML(df):
    """Create all possible baselines as well as tuned models and return the best one with respect to "MAE" score.

    Keyword argument:
    df -- dataframe of input data
    """

    reg = setup(data=df, target='demand', session_id=123,  normalize=True, fold=2, numeric_features=config["numeric"],
                  ordinal_features={"item_id": list(df["item_id"].unique())},
                  data_split_shuffle=False, fold_strategy="timeseries", silent=True)
    all_compared = compare(models_chosen=["Linear Regression", "Light Gradient Boosting Machine"], show_metrics=False)

    with st.spinner("Tuning..."):
        all_tuned = [tune_model(i, n_iter=3, verbose=True, optimize="MAE") for i in all_compared]

    final = automl(optimize="MAE")
    final_created = create_model(final)
    met = pull()

    # getting name of final estimator to display later
    est_name = str(final_created).split("(")[0]
    met.drop(["R2", "MSE", "RMSE"], axis=1, inplace=True)
    met.reset_index(inplace=True, drop=True)
    met = met.iloc[-2]
    met = pd.DataFrame(met.values, columns=[est_name], index=met.index)
    met = met.transpose()
    met = met.style.set_properties(**{'background-color': "#4DBFD9"}, subset=['MAE'])

    st.write(f"{est_name} is the best model with respect to MAE score and is chosen for predictions")
    st.dataframe(met)

    return final

def predict(final, model_selected):
    """Return dataframe with predictions of the model chosen by user.

    Keyword arguments:
    final -- a list of trained (tuned) model objects
    model_selected -- name of algorithm chosen by user to make predictions
    """

    for i in final:
        if isinstance(i, config["mod_class_dict"][model_selected]):
            st.write("is DT REgression")

    pred_button = st.sidebar.button("Predict")

    if not pred_button:
        st.stop()

    predictions = predict_model(final)
    fin_met = pull()
    st.dataframe(fin_met)