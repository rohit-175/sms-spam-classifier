# SMS Spam Classifier

## Overview
The SMS Spam Classifier is a machine learning project designed to detect and filter out spam messages from legitimate (ham) messages. Using a Naive Bayes classifier, the system achieves remarkable accuracy and precision. The classifier is demonstrated via a Jupyter Notebook, where users can test it with custom inputs and view the corresponding predictions.

## Key Features
- **97% Accuracy**: Highly accurate predictions for distinguishing spam messages from ham.
- **100% Precision**: Zero false positives, ensuring that no legitimate message is mislabeled as spam.
- **Efficient Text Processing**: Utilizes TF-IDF vectorization to extract meaningful features from text.
- **Interactive Testing**: Users can input SMS messages directly in the Jupyter Notebook and receive instant results.

## Project Details
### Model
- **Algorithm**: Naive Bayes Classifier
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Feature Limit**: 3000 features for optimal performance

### Dataset
The classifier was trained and tested on a publicly available dataset of SMS messages labeled as spam or ham. The dataset was preprocessed for tokenization, stopword removal, and vectorization. https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Requirements
To run the project, ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- Required libraries:
  - pandas
  - numpy
  - scikit-learn

Install the required Python libraries using:
```bash
pip install pandas numpy scikit-learn
```

## How to Use
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd sms-spam-classifier
   ```
2. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the `sms_spam_classifier.ipynb` file in Jupyter Notebook.
4. Run the notebook cells step-by-step to load the model, preprocess the data, and test the classifier.
5. Use the provided input cell to enter a custom SMS message and see if it’s classified as spam or ham.

Feel free to contribute, report issues, or suggest improvements!

