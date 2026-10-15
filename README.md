# Phishing Email Detection Model

A beginner-friendly machine-learning mini-project using **Scikit-learn**. It combines:
- **TF-IDF** text features (words and two-word phrases)
- **URL and keyword indicators** (URL count, IP-based URLs, urgency, credential requests, reward language, and more)
- **Logistic Regression** classification
- Accuracy, classification report, and a saved confusion-matrix plot
- An interactive prompt to test new email text

## 1. Requirements

Install Python 3.10+ and open a terminal in this project folder.

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
python phishing_detector.py
```

The program saves `confusion_matrix.png` in the project folder and then lets you paste an email to classify. Type `quit` to exit.

## 3. Use your own dataset (recommended)

Create `data/emails.csv` with at least these two columns:

```csv
text,label
"Urgent: verify your account using this link",phishing
"Please find the meeting agenda attached",safe
```

Labels can be `phishing` / `safe`; common alternatives like `spam` / `ham` and `1` / `0` are also accepted. Include many examples from both classes. Keep the test set separate from training through the included split, and avoid duplicate or near-duplicate emails across splits.

If `data/emails.csv` is absent, the project uses a tiny built-in **demonstration dataset** so you can run it immediately. Metrics from that small dataset are only a pipeline demonstration and must not be reported as real-world performance.

## 4. Project workflow

1. Load labeled emails.
2. Split into training and testing sets.
3. Convert email text into TF-IDF vectors.
4. Extract URL and social-engineering indicators.
5. Train Logistic Regression.
6. Evaluate on the held-out test set.
7. Show a confusion matrix and classify new input.

## 5. Important limitations

- No model can guarantee an email is safe. Treat predictions as a warning signal, not a security decision.
- Performance depends on dataset size, quality, language, and how representative the examples are.
- This starter project does not parse full `.eml` headers, attachments, sender reputation, or live URLs.
- Do not click links in suspicious messages. Use a safe, isolated environment for security testing.

## Suggested dataset sources

Look for reputable public phishing-email datasets from academic repositories or Kaggle. Check the license, labels, and provenance before use. Never include personal inbox data without removing sensitive information and obtaining appropriate permission.
