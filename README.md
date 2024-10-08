# Tool for Binary Age Prediction of the Domestic Feline

This is a demo repository for my dissertation: "_A WHISKER OF TRUTH: A DEEP LEARNING PIPELINE FOR AGE PREDICTION FROM VOCALISATIONS OF THE DOMESTIC FELINE_"

Basic UI for demonstration purposes. This tool is designed for predicting the age group (kitten, senior) of samples based on pre-extracted embeddings. The model is trained on a full dataset, with a few samples held back for performance demonstration.

In a production environment, support for uploading wav files to extract embeddings in real-time will be added. 


## Demo Video

A demo video is included to showcase the functionality of the tool. The actual class column is included for performance demonstration. In a production environment, this column would not be present. Best viewed with sound on for commentary. 


https://github.com/aster-droide/age-prediction-demo-binary/assets/105680030/2664fc0b-490a-4242-93b4-4aaa71ecaac1


## Usage

The model build (`cat_age_model.keras`) is packaged and included in this repo. The StandardScaler scaled on the training data, is included as well (`scaler_full.pkl`). Embeddings will be scaled to expected format after which predictions are made. 

To run the demo:

1. Start the application by running `binary-demo-app.py`
2. Navigate to the port `http://127.0.0.1:5001`
3. Select "Choose File" to upload a CSV file with embeddings; samples are included in this repo
4. Click on "Predict" to view the predictions and probabilities for each sample
