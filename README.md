
# SPAM DETECTION SYSTEM 

Spam detection is the process of identifying and filtering out unwanted or unsolicited messages, such as emails, SMS, or social media messages, that are often sent in bulk. These messages typically contain advertisements, phishing attempts, or malicious content. Project Overview: The Spam Detection System is an AI-ML model designed to detect spam messages. It uses a combination of XGBoost, logistic regression, and Naive Bayes algorithms, with a TF-IDF vectorizer for text analysis.Model Deployment: The model can be accessed via a website and Google Colab for quick analysis and performance statistics.

-------------------------------------------------------------------------------------------------

## Model Deployment
**Checkout Google Collab** <br>-[Open file on Colab ]https://colab.research.google.com/drive/1-MPzSPxzApVlQ6vgAESqlyvwHSq7suMV?usp=sharing&pli=1&authuser=1<br>

-------------------------------------------------------------------------------------------------

### Dataset Pre-Processing
### Key Concepts in Spam Detection
### *Data Collection:
Collecting a large dataset of messages labeled as spam or not spam.
### *Preprocessing:
Cleaning and preparing the data for analysis. This includes removing irrelevant information and converting text into a format that can be processed by machine learning algorithms.
### *Feature Extraction:
Identifying characteristics or patterns in the text that can help distinguish spam from legitimate messages. Common features include word frequency, presence of certain keywords, and message length.
### *Model Training:
Using machine learning algorithms to train a model on the labeled dataset. Popular algorithms include Naive Bayes, Support Vector Machines (SVM), and Random Forest.
### *Evaluation:
Assessing the model’s performance using metrics like accuracy, precision, recall, and F1-score to ensure it effectively identifies spam.
### *Deployment:
Implementing the trained model in a real-world environment to automatically classify incoming messages as spam or not spam.

-------------------------------------------------------------------------------------------------
### 1. **Import Libraries**
   You are importing several important libraries at the beginning of the code:
   - `pandas`, `numpy`: For data manipulation and numeric operations.
   - `re`: For regular expressions, used for text cleaning.
   - `nltk`: Used for natural language processing, especially stopwords removal and stemming.
   - `TfidfVectorizer`: Converts text to a vector representation based on term frequency-inverse document frequency (TF-IDF).

### 2. **Load Datasets**
   You are loading two datasets:
   - `spam.csv`: Contains a labeled dataset of spam or ham messages (`dataset_1`).
   - For `dataset_1`, you replace the labels `FAKE` with `1` and `REAL` with `0`.

### 3. **Concatenate Datasets**
   - Both datasets are combined into a single DataFrame `dataset` using `pd.concat`. This allows you to work with one dataset, combining the ham or spam messages.

### 4. **Stemming Function**
   - You define a `stemming()` function that:
     - Removes non-alphabetical characters.
     - Converts the text to lowercase.
     - Splits the text into words.
     - Removes stopwords (common words like "the", "is").
     - Stems words (e.g., "running" becomes "run") using `PorterStemmer`.
     - Rejoins the words into a cleaned text string.

### 6. **Apply Stemming**
   - You apply the `stemming()` function to the `text` column of the `dataset` to preprocess all the text data.

### 7. **Vectorize Text**
   - `TfidfVectorizer` is used to convert the preprocessed text into a numerical format (TF-IDF vectors) for model training.
   - You `fit_transform` the vectorizer on the `X` dataset (the text column), which learns the vocabulary and transforms the text into vectors.

-------------------------------------------------------------------------------------------------

## Approach
1. **Data Loading and Preprocessing**:
   - Two datasets are loaded: one containing e-mail/messages with labels (fake or real) and another with additional text data.
   - Unnecessary columns were removed
   - Missing values are filled with empty spaces, and labels are converted to binary (FAKE = 1, REAL = 0).
   - A **stemming** function is applied to clean and preprocess the text, removing unwanted characters and stopwords, and reducing words to their root form.
   - Both datasets are merged into one and the text is transformed using the **TF-IDF vectorizer** to convert text into numerical features.

2. **Train-Test Split**:
   - The data is split into training (80%) and testing (20%) sets using **train_test_split**.
   
3. **Model Training**:
   - Three machine learning models are trained on the training set:
     - **Logistic Regression**
     - **XGBoost**
     - **Naive Bayes**
   - Each model is evaluated on the training and testing sets using **accuracy_score** to calculate prediction accuracy.

4. **Output**:
   - The program displays the prediction (REAL or FAKE) for the input e-mail/messages data and provides the accuracy of the models on the test data.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------     
## Workflow

1. Data Preparation
2. Pre Processing
3. Vectorization
4. Splitting the data
5. Training The models
6. Testing with manual Inputs

------------------------------------------------------------------------------------------------------------------------------------------------------------
## ACKNOWLEGEMENT

1) Shivam Pandey: Team Lead   

2)Rudra Kanojiya : ML Developer  

3)Sneha Dubey : GitHub Manager and FrontEnd Developer  

4)Jaideep Khare* : GitHub Manager and Algoman  

