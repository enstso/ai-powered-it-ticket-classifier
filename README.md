# Intelligent IT Ticket Classification

This README summarizes the project objective, expected implementation, and current evaluation observations. The project is a Python application that classifies IT support tickets written in natural language into one of four categories:

- `network`
- `access`
- `software`
- `security`

The goal is to help route support requests to the right team faster, while keeping human validation when the model is not confident enough.

## Project Context

An IT support team receives many tickets every day, for example:

- "I can no longer connect to the VPN."
- "My password has expired."
- "The application closes when it starts."
- "I received a suspicious email."

These tickets are usually assigned manually to the correct support team. Manual routing takes time and can lead to classification mistakes. This project explores whether a machine learning model can assist with that routing.

## Objectives

The application is designed to:

- load and analyze a ticket dataset with Pandas;
- clean duplicated, missing, and inconsistent data;
- preprocess ticket text with NLTK;
- train a multiclass classification model with scikit-learn;
- compare the trained model against a baseline;
- evaluate the model with accuracy, precision, recall, F1-score, and a confusion matrix;
- expose predictions through a Flask API;
- save the trained model with `joblib` or `pickle`;
- run inside Docker;
- request human validation when prediction confidence is too low.

## Tech Stack

- Python 3.11+
- Pandas
- scikit-learn
- NLTK
- Flask
- Docker
- pytest
- joblib or pickle

## Expected Project Structure

```text
.
├── data/
│   ├── tickets.csv
│   └── clean_tickets.csv
├── models/
│   └── ticket_classifier.joblib
├── reports/
│   └── prediction_errors.csv
├── src/
│   ├── app.py
│   ├── data_analysis.py
│   ├── preprocessing.py
│   └── train_model.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Dataset

The dataset must contain at least the following columns:

```csv
ticket_id,text,category,priority
```

The required categories are:

- `network`
- `access`
- `software`
- `security`

The dataset should include several dozen examples per category, different formulations of similar problems, uppercase/lowercase variations, punctuation variations, a few missing values, and a few duplicates. These imperfections are intentional because they make the data cleaning step more realistic.

## Data Analysis

The script `src/data_analysis.py` is expected to inspect the dataset and display:

- number of rows and columns;
- data types;
- missing values;
- duplicated rows;
- category distribution;
- priority distribution;
- average ticket length;
- most frequent category.

This first analysis helps identify whether the dataset is balanced enough and whether some categories may dominate the predictions.

## Data Cleaning

The cleaning step removes unreliable or inconsistent rows before training.

The `clean_dataset(df)` function should:

- remove duplicates;
- remove rows with missing text or category;
- normalize categories to lowercase;
- remove unnecessary spaces;
- keep only authorized categories;
- add a `text_length` column;
- save the cleaned dataset to `data/clean_tickets.csv`.

## NLP Preprocessing

The function `preprocess_text(text: str) -> str` applies basic NLTK preprocessing:

- lowercase conversion;
- tokenization;
- punctuation removal;
- stop word removal;
- empty token removal;
- reconstruction of the cleaned text.

Example:

```text
Input:  "Je ne peux plus me connecter au VPN !"
Output: "peux connecter vpn"
```

The function must also handle empty text safely.

## Modeling Approach

The project compares a baseline model with trained machine learning models.

### Baseline

The baseline uses:

```python
DummyClassifier(strategy="most_frequent")
```

This model always predicts the most frequent class. The final model must perform clearly better than this baseline.

### Train/Test Split

The dataset is split into training and testing sets:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set is only used for evaluation and must not be used during training.

### Model Variants

Two model variants are compared:

- **Model A:** TF-IDF applied to lightly cleaned raw text.
- **Model B:** NLTK preprocessing followed by TF-IDF.

The classifier is based on a scikit-learn pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

## Evaluation Summary

The model is evaluated with:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

Based on the current analysis notes:

- The model performs above the baseline/reference threshold of `0.25`.
- The category with the best recall is `network`.
- The category with the lowest precision is also `network`.
- The most frequent confusion involves `network`, followed by `software`.
- NLTK preprocessing does not clearly improve the model in the current experiment.

This means the model identifies many `network` tickets, but it may also over-predict this category. In other words, recall is strong for `network`, but precision is weaker because some tickets from other categories are incorrectly classified as `network`.

## Error Analysis

The project exports prediction errors to:

```text
reports/prediction_errors.csv
```

Expected columns:

```csv
text,expected_category,predicted_category
```

The manual error analysis suggests several possible causes:

- The French word `connexion` appears to be strongly associated with the `network` category.
- This may cause the model to predict `network` even when the ticket is actually about access, authentication, or security.
- The preprocessing may be too aggressive and may remove useful context.
- Stemming or lemmatization may transform words incorrectly, especially with French text.
- Some stop words may actually carry useful meaning for ticket classification.
- The dataset may be too small, especially if the first experiments were done on around fifty rows.
- Some tickets may be ambiguous or mention several issues at the same time.

To investigate this further, it is useful to compare the text before and after preprocessing step by step. This helps identify whether important words are removed or transformed in a way that hurts prediction quality.

## Can the Model Fully Automate Ticket Routing?

Not yet.

The current model performs better than the baseline, and some results are encouraging. However, the error analysis shows that more adjustments are needed before using it for full automation.

Recommended improvements:

- add more labeled training examples;
- balance the dataset across categories;
- improve French preprocessing;
- test lemmatization instead of aggressive stemming;
- review the stop word list;
- inspect the most frequent confusion cases;
- tune the confidence threshold;
- keep human validation for low-confidence predictions.

For now, the model should be used as an assistance tool, not as a fully autonomous routing system.

## Flask API

The Flask API should expose an endpoint that receives a ticket text and returns a predicted category.

Example request:

```http
POST /predict
Content-Type: application/json

{
  "text": "Je ne peux plus me connecter au VPN"
}
```

Example response:

```json
{
  "category": "network",
  "confidence": 0.82,
  "requires_human_validation": false
}
```

If confidence is too low, the API should return a response indicating that human validation is required:

```json
{
  "category": "network",
  "confidence": 0.43,
  "requires_human_validation": true
}
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data analysis:

```bash
python src/data_analysis.py
```

Train the model:

```bash
python src/train_model.py
```

Start the Flask API:

```bash
python src/app.py
```

Run tests:

```bash
pytest
```

## Docker

Build the Docker image:

```bash
docker build -t it-ticket-classifier .
```

Run the container:

```bash
docker run -p 5000:5000 it-ticket-classifier
```

The API should then be available at:

```text
http://localhost:5000
```

## Conclusion

This project shows that machine learning can help classify IT support tickets and reduce manual routing work. The trained model performs better than the baseline, especially for the `network` category. However, the current results are not strong enough for full automation.

The next priority is to improve data quality, increase the size of the dataset, refine French NLP preprocessing, and keep a human-in-the-loop workflow for uncertain predictions.
