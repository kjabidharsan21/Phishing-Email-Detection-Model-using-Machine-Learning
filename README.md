## Phishing Email Detection Model using Machine Learning

This project focuses on detecting phishing emails using Machine Learning and Python. Phishing emails are designed to trick users into sharing sensitive information, such as passwords, banking details, and account credentials, by pretending to be legitimate messages.

The model uses **Scikit-learn** to analyze email text, identify suspicious keywords, and examine URL-related features to classify messages as **Phishing** or **Safe**. It uses TF-IDF for text feature extraction and Logistic Regression for classification.

The project also evaluates the model using accuracy, a classification report, and a confusion matrix to understand its performance. Users can enter new email messages and check the model's predictions.

### Key Features
- Classifies emails as phishing or safe.
- Extracts meaningful features from email text using TF-IDF.
- Analyzes suspicious URLs and common phishing-related keywords.
- Uses Logistic Regression for classification.
- Displays accuracy, classification metrics, and a confusion matrix.
- Allows users to test new email messages interactively.

### Technologies Used
- Python
- Scikit-learn
- Pandas
- Matplotlib
- TF-IDF Vectorization
- Logistic Regression

### Learning Outcomes
Through this project, I explored text preprocessing, feature extraction, supervised machine learning, model evaluation, and phishing email detection. This project helped me understand how machine learning can be applied to a real-world cybersecurity problem.

**Note:** The current version includes a small demonstration dataset. Real-world performance depends on training and evaluating the model with a larger, representative dataset.
