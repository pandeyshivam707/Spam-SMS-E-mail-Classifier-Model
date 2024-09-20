
**SPAM DETECTION SYSTEM :-** <br>
Spam detection is the process of identifying and filtering out unwanted or unsolicited messages, such as emails, SMS, or social media messages, that are often sent in bulk. These messages typically contain advertisements, phishing attempts, or malicious content. Project Overview: The Spam Detection System is an AI-ML model designed to detect spam messages. It uses a combination of XGBoost, logistic regression, and Naive Bayes algorithms, with a TF-IDF vectorizer for text analysis.Model Deployment: The model can be accessed via a website and Google Colab for quick analysis and performance statistics.

-------------------------------------------------------------------------------------------------

**How to Pre-Processing datasets**<br>
**Data Collection:** <br>
Collecting a large dataset of messages labeled as spam or not spam.<br>
**Preprocessing:** <br>
Cleaning and preparing the data for analysis. This includes removing irrelevant information and converting text into a format that can be processed by machine learning algorithms.<br>
**Feature Extraction:** <br>
Identifying characteristics or patterns in the text that can help distinguish spam from legitimate messages. Common features include word frequency, presence of certain keywords, and message length. <br>
**Model Training:** <br>
Using machine learning algorithms to train a model on the labeled dataset. Popular algorithms include Naive Bayes, Support Vector Machines (SVM), and Random Forest. <br>
**Evaluation:** <br>
Assessing the model’s performance using metrics like accuracy, precision, recall, and F1-score to ensure it effectively identifies spam. <br>
**Deployment:** <br>
Implementing the trained model in a real-world environment to automatically classify incoming messages as spam or not spam.

-------------------------------------------------------------------------------------------------
## How to implemet ???
Data Loading and Preprocessing:
   - Two datasets are loaded: one containing e-mail/messages with labels (fake or real) and another with additional text data. Unnecessary columns were removedMissing values are filled with empty spaces, and labels are converted to binary (FAKE = 1, REAL = 0).A stemming function is applied to clean and preprocess the text, removing unwanted characters and stopwords, and reducing words to their root form.Both datasets are merged into one and the text is transformed using the TF-IDF vectorizer to convert text into numerical features.
Train-Test Split:The data is split into training (80%) and testing (20%) sets using train_test_split.
Output:The program displays the prediction (REAL or FAKE) for the input e-mail/messages data and provides the accuracy of the models on the test data.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------     
## Workflow

1. Data Preparation
2. Pre Processing
3. Vectorization
4. Splitting the data
5. Training The models
6. Testing with manual Inputs

--------------------------------------------------------------------------------------------------------------------------------------------------------

## Model Deployment
**Checkout Google Collab** https://colab.research.google.com/drive/1-MPzSPxzApVlQ6vgAESqlyvwHSq7suMV?usp=sharing&pli=1&authuser=1<br>

------------------------------------------------------------------------------------------------------------------------------------------------------------

