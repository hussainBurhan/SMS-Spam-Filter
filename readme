# SMS Filter Project

## Overview
This repository contains code for a simple SMS (Short Message Service) filter project, specifically designed to classify messages as either spam or not spam (ham). The project includes a Streamlit web application for user interaction and a machine learning model trained on a dataset of SMS messages.

## Project Structure
- **app.py:** Streamlit application code that allows users to input a message and receive a classification prediction.
- **main.py:** Main script for data preprocessing, exploratory data analysis (EDA), and machine learning model building.
- **SMSSpamCollection:** Dataset file containing SMS messages labeled as spam or ham.
- **vectorizer.pkl:** Pickle file containing the trained TF-IDF vectorizer used for text representation.
- **model.pkl:** Pickle file containing the trained Multinomial Naive Bayes classifier for spam classification.

## Instructions for Running the Streamlit App
1. Install the required Python libraries using `pip install -r requirements.txt`.
2. Run the Streamlit app by executing `streamlit run app.py` in the terminal.
3. Access the app through the provided URL (usually http://localhost:8501) in your web browser.
4. Enter a message in the text area and click the "Predict" button to see the classification result.

## Data Preprocessing and EDA
- The `main.py` script reads the SMS dataset, performs data cleaning, encodes the target variable, and conducts exploratory data analysis.
- Descriptive statistics, visualizations, and word clouds are generated to gain insights into the characteristics of spam and ham messages.

## Model Building
- The script builds a machine learning model using the Multinomial Naive Bayes algorithm for spam classification.
- The TF-IDF vectorizer is used for text representation, and the model is trained on the preprocessed SMS messages.
- The trained model and vectorizer are saved as `model.pkl` and `vectorizer.pkl` for future use.

## Important Files
- **app.py:** Streamlit application for user interaction.
- **main.py:** Script for data preprocessing, exploratory data analysis, and model building.
- **vectorizer.pkl:** Trained TF-IDF vectorizer.
- **model.pkl:** Trained Multinomial Naive Bayes classifier.

## Dependencies
- Python 3.x
- Streamlit
- NLTK
- scikit-learn
- pandas
- matplotlib
- seaborn
- wordcloud

## Acknowledgments
- The dataset used in this project is sourced from the SMS Spam Collection available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
- Special thanks to the contributors of NLTK, scikit-learn, Streamlit, and other open-source libraries used in this project.

Feel free to explore the code, run the Streamlit app, and contribute to further improvements! If you encounter any issues or have suggestions, please open an issue in the repository.
