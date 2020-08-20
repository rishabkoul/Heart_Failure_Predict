# Core Pkgs
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection
import seaborn as sns
import streamlit as st

# EDA pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ML pkgs

st.set_option('deprecation.showfileUploaderEncoding', False)


def main():
    st.write("""
# Heart Failure Prediction App

This app predicts the **Heart Failure** for a patient.

Data obtained from [here](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5).
""")
    activities = ["EDA", "Plot", "Model Building", "About"]

    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

        # data = st.file_uploader("Upload Dataset", type=["csv", "txt"])
        # if data is not None:
        #     df = pd.read_csv(data)
        st.dataframe(df)

        if st.checkbox("Show shape"):
            st.write(df.shape)

        if st.checkbox("Show columns"):
            all_columns = df.columns.to_list()
            st.write(all_columns)

        if st.checkbox("Select Columns To Show"):
            selected_columns = st.multiselect(
                "Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Show summary"):
            st.write(df.describe())

        if st.checkbox("Show value counts"):
            st.write(df.iloc[:, -1].value_counts())

    elif choice == 'Plot':
        st.subheader("Data Visualization")

        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        st.dataframe(df)

        if st.checkbox("Correlation with Seaborn"):
            corr = df.corr()
            st.write(sns.heatmap(corr))
            st.pyplot()

        if st.checkbox("Pie Chart"):
            all_columns = df.columns.to_list()
            colums_to_plot = st.selectbox(
                "Select 1 column to plot", all_columns)
            pie_plot = df[colums_to_plot].value_counts().plot.pie()
            st.write(pie_plot)
            st.pyplot()

        all_columns = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Plot", [
                                    "area", "bar", "line", "hist", "box", "kde"])
        selected_columns_names = st.multiselect(
            "selct Columns To plot", all_columns)

        if st.button("Generate Plot"):
            st.success("Generating Customizable Plot of {} for {}".format(
                type_of_plot, selected_columns_names))

            # Plot by streamlit
            if type_of_plot == "area":
                cust_data = df[selected_columns_names]
                st.area_chart(cust_data)

            elif type_of_plot == "bar":
                cust_data = df[selected_columns_names]
                st.bar_chart(cust_data)

            elif type_of_plot == "line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            # Custom plot
            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

    elif choice == 'Model Building':
        st.subheader("Building ML Model")

        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        st.dataframe(df)

        # Model building
        X = df.iloc[:, 0:-1]
        Y = df.iloc[:, -1]
        seed = 7

        # Model
        models = []
        models.append(("LR", LogisticRegression()))
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(probability=True, gamma="auto")))
        models.append(('RFC', RandomForestClassifier(n_estimators=100)))
        models.append(('GBC', GradientBoostingClassifier()))

        # evaluate each model in turn

        # List
        model_name = []
        model_mean = []
        model_std = []
        all_models = []
        scoring = 'accuracy'

        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(
                model, X, Y, cv=kfold, scoring=scoring)
            model_name.append(name)
            model_mean.append(cv_results.mean())
            model_std.append(cv_results.std())

            accuracy_results = {"model_name": name, "model_accuracy": cv_results.mean(
            ), "standard_deviation": cv_results.std()}
            all_models.append(accuracy_results)

        if st.checkbox("Metrics as Table"):
            st.dataframe(pd.DataFrame(
                zip(model_name, model_mean, model_std), columns=['Model Name', 'Model Accuracy', 'Standard Deviation']))

        if st.checkbox("Metrics as Json"):
            st.json(all_models)

        def user_input_features():
            age = st.sidebar.slider('Age of the patient(Years)', 40, 95, 50)
            anaemia = st.sidebar.selectbox(
                'Anaemia-Decrease of red blood cells or hemoglobin(True-1, False-0)', (1, 0))
            creatinine_phosphokinase = st.sidebar.slider(
                'Creatinine phosphokinase-Level of the CPK enzyme in the blood(mcg/L)', 23, 7861, 3300)
            diabetes = st.sidebar.selectbox(
                'Diabetes-If the patient has diabetes(True-1, False-0)', (1, 0))
            ejection_fraction = st.sidebar.slider(
                'Ejection fraction-Percentage of blood leaving', 14, 80, 30)
            high_blood_pressure = st.sidebar.selectbox(
                'High blood pressure-If a patient has hypertension(True-1, False-0)', (1, 0))
            platelets = st.sidebar.slider(
                'Platelets-Platelets in the blood(kiloplatelets/mL)', 25100, 850000, 40000)
            serum_creatinine = st.sidebar.slider(
                'Serum creatinine-Level of creatinine in the blood(mg/dL)', 0.5000, 9.4000, 1.2000)
            serum_sodium = st.sidebar.slider(
                'Serum sodium-Level of sodium in the blood(mEq/L)', 113, 148, 120)
            sex = st.sidebar.selectbox(
                'Sex-Woman or Man(Man-1,Women-0)', (1, 0))
            smoking = st.sidebar.selectbox(
                'Smoking-If the patient smokes(True-1, False-0)', (1, 0))
            time = st.sidebar.slider(
                'Time-Follow-up period(Days)', 4, 285, 100)
            data = {'age': age,
                    'anaemia': anaemia,
                    'creatinine_phosphokinase': creatinine_phosphokinase,
                    'diabetes': diabetes,
                    'ejection_fraction': ejection_fraction,
                    'high_blood_pressure': high_blood_pressure,
                    'platelets': platelets,
                    'serum_creatinine': serum_creatinine,
                    'serum_sodium': serum_sodium,
                    'sex': sex,
                    'smoking': smoking,
                    'time': time
                    }
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()
        if input_df is not None:
            st.dataframe(input_df)
        for name, model in models:
            model.fit(X, Y)
            prediction_proba = model.predict_proba(input_df)
            st.write('{} Predictions:'.format(name))
            st.write(prediction_proba)
            index = np.argmax(prediction_proba)
            if index == 0:
                st.write('Not Dead')
            else:
                st.write('Dead')

    elif choice == 'About':
        st.subheader("About")
        st.write("Made By Rishab Koul with the Streamlit Library")


if __name__ == '__main__':
    main()
