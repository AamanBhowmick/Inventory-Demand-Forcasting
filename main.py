from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import seaborn as sns

encoder = LabelEncoder()

local_server = True
app = Flask(__name__)

app.secret_key = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = "./dataset/"
app.config['CSV_NAME'] = ""

permissions = 0o700
os.chmod(app.config['UPLOAD_FOLDER'], permissions)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        app.config['CSV_NAME'] = ""
        f = request.files['file_name']
        # print(f.filename)
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        app.config['CSV_NAME'] = app.config['CSV_NAME'] + f.filename
        print(app.config['CSV_NAME'])
        return redirect('/report')
    return render_template('upload.html')


@app.route('/report', methods=['GET', 'POST'])
def report():
    permissions = 0o700
    os.chmod(app.config['UPLOAD_FOLDER'], permissions)
    dataset_name = app.config['UPLOAD_FOLDER'] + app.config['CSV_NAME']
    # dataset_name_f = open(dataset_name, 'w')
    df = pd.read_csv(dataset_name)

    # value = df.loc[2, 'Product Code']

    df_new = pd.read_csv(dataset_name)
    first_row = df_new.columns
    features = []
    for i in first_row:
        features.append(i)
    features.remove('Date')
    features.remove('Sales Quantity')
    features.remove('Sales Revenue')
    features.remove('Product Code')
    df_new.dropna(inplace=True)
    df_new["Date"] = pd.to_datetime(df_new["Date"])
    df_new["Year"] = df_new["Date"].dt.year
    df_new["Month"] = df_new["Date"].dt.month
    df_new["Day"] = df_new["Date"].dt.day
    df_new["Sales Quantity"] = df_new["Sales Quantity"].astype(int)

    for i in features:
        df_new[i] = df_new[i].astype(str)

    for i in features:
        df_new[i] = encoder.fit_transform(df_new[i])

    features_dict = {}
    for i in features:
        n = df_new[i].nunique()
        features_dict[i] = n

    categories = df_new["Product Code"].unique()

    p_code = []
    for i in categories:
        p_code.append(i)
    
    category_predictions_df_xgb = pd.DataFrame(
        columns=['Product Code', 'Actual Quantity', 'Predicted Quantity'])
    category_future_predictions_df_xgb = pd.DataFrame(
        columns=['Product Code', 'Future Predicted Quantity'])

    for category in categories:
        category_data = df_new[df_new["Product Code"] == category]

        X = category_data.drop(['Sales Quantity', 'Sales Revenue'], axis=1)
        y = category_data['Sales Quantity']

        X_train_d, X_test_d, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train = X_train_d.drop(['Date'], axis=1)
        X_test = X_test_d.drop(['Date'], axis=1)

        xg_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xg_reg.fit(X_train, y_train)

        last_date = category_data['Date'].max()
        future_dates = pd.date_range(last_date, periods=30, freq='D')[1:]
        future_X = pd.DataFrame({'Date': future_dates})

        for i in range(0, 29):
            future_X["Product Code"] = category
            for key, value in features_dict.items():
                future_X.loc[i, key] = random.randint(0, value-1)
        future_X["Year"] = future_X["Date"].dt.year
        future_X["Month"] = future_X["Date"].dt.month
        future_X["Day"] = future_X["Date"].dt.day

        future_X.set_index('Date', inplace=True)
        future_y_pred = xg_reg.predict(future_X)
        future_results_df = pd.DataFrame(
            {'Predicted': future_y_pred}, index=future_X.index)

        results_df = pd.DataFrame(
            {'Date': X_test_d['Date'], 'Actual': y_test, 'Predicted': xg_reg.predict(X_test)})
        last_60_days = results_df['Date'] > (
            results_df['Date'].max() - pd.DateOffset(days=60))
        results_df = results_df[last_60_days]
        grouped_df = results_df.groupby(['Date']).mean().reset_index()

        predicted_quantity_future = future_results_df['Predicted'].sum().round(0)
        actual_quantity = grouped_df['Actual'].sum().round(0)
        predicted_quantity = grouped_df['Predicted'].sum().round(0)

        category_predictions_df_xgb = category_predictions_df_xgb.append(
            {'Product Code': category, 'Actual Quantity': actual_quantity, 'Predicted Quantity': predicted_quantity}, ignore_index=True)
        category_future_predictions_df_xgb = category_future_predictions_df_xgb.append(
            {'Product Code': category, 'Future Predicted Quantity': predicted_quantity_future}, ignore_index=True)
        
        category_future_predictions_df_xgb_sorted = category_future_predictions_df_xgb.sort_values('Future Predicted Quantity', ascending=False)

        # get the sorted 'Product Code' column
        # product_code_sorted = category_future_predictions_df_xgb_sorted['Product Code']
        
        plt.figure(figsize=(10, 8))
        plot = sns.barplot(data=category_future_predictions_df_xgb, x='Product Code', y='Future Predicted Quantity')
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        plot_path = 'static/images/plot.png'    
        plt.savefig(plot_path)

    return render_template("report.html", tables1=[df.head(10).to_html(classes='data', table_id='my-table1')], tables2=[df.tail(10).to_html(classes='data', table_id='my-table1')], tables3=[category_future_predictions_df_xgb_sorted.to_html(classes='data')], plot_path = plot_path, p_code=p_code)

@app.route('/details', methods=['GET', 'POST'])
def report_details():
    permissions = 0o700
    os.chmod(app.config['UPLOAD_FOLDER'], permissions)
    dataset_name = app.config['UPLOAD_FOLDER'] + app.config['CSV_NAME']
    # dataset_name_f = open(dataset_name, 'w')
    df = pd.read_csv(dataset_name)

    # value = df.loc[2, 'Product Code']

    df_new = pd.read_csv(dataset_name)
    first_row = df_new.columns
    features = []
    for i in first_row:
        features.append(i)
    features.remove('Date')
    features.remove('Sales Quantity')
    features.remove('Sales Revenue')
    features.remove('Product Code')
    df_new.dropna(inplace=True)
    df_new["Date"] = pd.to_datetime(df_new["Date"])
    df_new["Year"] = df_new["Date"].dt.year
    df_new["Month"] = df_new["Date"].dt.month
    df_new["Day"] = df_new["Date"].dt.day
    df_new["Sales Quantity"] = df_new["Sales Quantity"].astype(int)

    for i in features:
        df_new[i] = df_new[i].astype(str)

    for i in features:
        df_new[i] = encoder.fit_transform(df_new[i])

    features_dict = {}
    for i in features:
        n = df_new[i].nunique()
        features_dict[i] = n

    categories = df_new["Product Code"].unique()

    p_code = []
    for i in categories:
        p_code.append(i)
        
    length = int(len(p_code))
    
    category_predictions_df_xgb = pd.DataFrame(
        columns=['Product Code', 'Actual Quantity', 'Predicted Quantity'])
    category_future_predictions_df_xgb = pd.DataFrame(
        columns=['Product Code', 'Future Predicted Quantity'])

    if request.method == 'POST':
        category = int(request.form.get('p_code'))
        print(category)
        category_data = df_new[df_new["Product Code"] == category]

        X = category_data.drop(['Sales Quantity', 'Sales Revenue'], axis=1)
        y = category_data['Sales Quantity']

        X_train_d, X_test_d, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train = X_train_d.drop(['Date'], axis=1)
        X_test = X_test_d.drop(['Date'], axis=1)

        xg_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xg_reg.fit(X_train, y_train)

        last_date = category_data['Date'].max()
        future_dates = pd.date_range(last_date, periods=30, freq='D')[1:]
        future_X = pd.DataFrame({'Date': future_dates})

        for i in range(0, 29):
            future_X["Product Code"] = category
            for key, value in features_dict.items():
                future_X.loc[i, key] = random.randint(0, value-1)
        future_X["Year"] = future_X["Date"].dt.year
        future_X["Month"] = future_X["Date"].dt.month
        future_X["Day"] = future_X["Date"].dt.day

        future_X.set_index('Date', inplace=True)
        future_y_pred = xg_reg.predict(future_X)
        future_results_df = pd.DataFrame(
            {'Predicted': future_y_pred}, index=future_X.index)

        results_df = pd.DataFrame(
            {'Date': X_test_d['Date'], 'Actual': y_test, 'Predicted': xg_reg.predict(X_test)})
        last_60_days = results_df['Date'] > (
            results_df['Date'].max() - pd.DateOffset(days=60))
        results_df = results_df[last_60_days]
        grouped_df = results_df.groupby(['Date']).mean().reset_index()

        predicted_quantity_future = future_results_df['Predicted'].sum().round(0)
        actual_quantity = grouped_df['Actual'].sum().round(0)
        predicted_quantity = grouped_df['Predicted'].sum().round(0)

        category_predictions_df_xgb = category_predictions_df_xgb.append(
            {'Product Code': category, 'Actual Quantity': actual_quantity, 'Predicted Quantity': predicted_quantity}, ignore_index=True)
        category_future_predictions_df_xgb = category_future_predictions_df_xgb.append(
            {'Product Code': category, 'Future Predicted Quantity': predicted_quantity_future}, ignore_index=True)
        
        # plt.figure(figsize=(10, 8))
        # plot = sns.lineplot(data=category_future_predictions_df_xgb, x='Product Code', y='Future Predicted Quantity')
        # plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        # plot_path = 'static/images/plot.png'    
        # plt.savefig(plot_path)
        
        plt.figure(figsize=(20, 8))
        plot2 = sns.lineplot(x=future_results_df.index, y='Predicted', data=future_results_df)
        plot_path2 = 'static/images/plota.png'    
        plt.savefig(plot_path2)
        
        plt.figure(figsize=(20, 8))
        plot3 = sns.lineplot(x='Date', y='Actual', data=grouped_df)
        plot_path3 = 'static/images/plotb.png'    
        plt.savefig(plot_path3)
        
    return render_template("report_details.html", tables1=[df.head(length).to_html(classes='data', table_id='my-table1')], p_code=p_code, plot_path2 = plot_path2, plot_path3 = plot_path3, predicted_quantity_future = predicted_quantity_future, category = category)

if __name__ == '__main__':
    app.run(debug=True)
