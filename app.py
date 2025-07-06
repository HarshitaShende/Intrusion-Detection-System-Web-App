from flask import Flask, render_template, request, send_file
import os
import pandas as pd
import joblib
import plotly.express as px
from utils.preprocessing import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = joblib.load('model/ids_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    df = None
    chart_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Read and preprocess
            df = pd.read_csv(filepath)
            X = preprocess_input(df)
            preds = model.predict(X)
            df['Prediction'] = preds
            df['Prediction'] = df['Prediction'].map({0: 'Normal', 1: 'Malicious'})

            # Save CSV for download
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'last_result.csv'), index=False)

            # Generate pie chart as image (no JS)
            pie_data = df['Prediction'].value_counts().reset_index()
            pie_data.columns = ['Class', 'Count']
            fig = px.pie(pie_data, names='Class', values='Count', title='Traffic Type Distribution')
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pie_chart.png')
            fig.write_image(chart_path)

    return render_template('index.html', df=df, chart_path=chart_path)

@app.route('/download')
def download():
    return send_file('static/uploads/last_result.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
