"""
Phishing Email Detection Model
------------------------------
A beginner-friendly Scikit-learn project that classifies emails as Phishing or Safe.

Quick start:
    pip install -r requirements.txt
    python phishing_detector.py

Dataset:
    Put a CSV file at data/emails.csv with columns:
        text,label
    Labels may be: phishing/safe, 1/0, or spam/ham.
    If no dataset is found, the program uses a small DEMONSTRATION dataset.
    The demonstration data is only for testing the pipeline, not for real-world use.

Optional URL features:
    The model combines TF-IDF text features with simple URL/link indicators.
"""

from pathlib import Path
import re
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, ConfusionMatrixDisplay,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
OUTPUT_PATH = ROOT / "confusion_matrix.png"

DEMO_DATA = [
    ("Urgent: verify your account now at http://secure-login-example.com or your account will be suspended", "phishing"),
    ("Your mailbox is full. Click here to keep your email active: http://mailbox-check-example.net", "phishing"),
    ("Congratulations, you won a prize. Submit your bank details to claim it immediately", "phishing"),
    ("We detected unusual activity. Confirm your password using this link now", "phishing"),
    ("Your payment failed. Update billing information at http://payment-update-example.org", "phishing"),
    ("Final warning: your account will be closed unless you verify your identity today", "phishing"),
    ("You have been selected for a cash reward. Send your account number to receive it", "phishing"),
    ("Please open the attachment and enable macros to view your invoice immediately", "phishing"),
    ("IT support needs your password urgently. Reply with your login credentials", "phishing"),
    ("Your package is on hold. Pay a small fee through this unfamiliar link", "phishing"),
    ("Can we move tomorrow's project meeting to 10 AM? The agenda is attached", "safe"),
    ("Hi team, please find the meeting notes from this morning attached", "safe"),
    ("Your order has shipped. Track it from the retailer's official website", "safe"),
    ("The monthly newsletter is available on our company intranet", "safe"),
    ("Reminder: your library book is due next Monday", "safe"),
    ("Thanks for your application. We will contact you after reviewing your resume", "safe"),
    ("Please review the updated project report before Friday", "safe"),
    ("Your appointment is confirmed for 2:30 PM on Thursday", "safe"),
    ("The class timetable has been uploaded to the college portal", "safe"),
    ("Here is the receipt for your recent purchase. Thank you for shopping with us", "safe"),
    ("Let's discuss the quarterly budget during our scheduled call", "safe"),
    ("Your password was changed successfully. Contact support if you did not make this change", "safe"),
    ("The team lunch is planned for Wednesday. Please confirm attendance", "safe"),
    ("Please find the lecture slides and assignment instructions attached", "safe"),
]

def normalize_label(value):
    """Map common dataset labels to phishing/safe."""
    s = str(value).strip().lower()
    if s in {"1", "phishing", "phish", "malicious", "spam", "fraud", "unsafe"}:
        return "phishing"
    if s in {"0", "safe", "legitimate", "ham", "benign", "not phishing", "not_phishing"}:
        return "safe"
    return None

def load_dataset():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Accept common column names.
        lower_map = {c.lower().strip(): c for c in df.columns}
        text_col = next((lower_map[c] for c in ("text", "email", "body", "content", "message") if c in lower_map), None)
        label_col = next((lower_map[c] for c in ("label", "class", "category", "target") if c in lower_map), None)
        if not text_col or not label_col:
            raise ValueError("data/emails.csv must contain a text/email/content column and a label/class column.")
        result = pd.DataFrame({"text": df[text_col].fillna("").astype(str),
                               "label": df[label_col].map(normalize_label)})
        result = result.dropna(subset=["label"])
        result = result[result["text"].str.strip().ne("")]
        if result["label"].nunique() < 2:
            raise ValueError("Dataset must contain both phishing and safe examples.")
        print(f"Loaded {len(result)} labeled emails from {DATA_PATH}")
        return result, False

    print("No data/emails.csv found. Using small DEMONSTRATION data.")
    print("For meaningful evaluation, replace it with a larger labeled email dataset.")
    return pd.DataFrame(DEMO_DATA, columns=["text", "label"]), True

class EmailURLFeatures(BaseEstimator, TransformerMixin):
    """Extract lightweight URL and social-engineering indicators from email text."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for item in X:
            text = str(item)
            lower = text.lower()
            urls = re.findall(r"(?:https?://|www\.)\S+", text, flags=re.I)
            rows.append({
                "url_count": len(urls),
                "has_ip_url": int(bool(re.search(r"https?://(?:\d{1,3}\.){3}\d{1,3}", text, re.I))),
                "has_shortener": int(bool(re.search(r"bit\.ly|tinyurl\.com|t\.co|goo\.gl", lower))),
                "has_https": int("https://" in lower),
                "has_http": int("http://" in lower),
                "url_has_at": int(any("@" in u for u in urls)),
                "urgent_word_count": sum(lower.count(w) for w in ["urgent", "immediately", "final warning", "act now"]),
                "credential_word_count": sum(lower.count(w) for w in ["password", "verify", "login", "credentials", "account"]),
                "money_reward_word_count": sum(lower.count(w) for w in ["prize", "cash", "reward", "won", "claim"]),
                "threat_word_count": sum(lower.count(w) for w in ["suspended", "closed", "blocked", "unusual activity"]),
                "has_attachment_action": int(any(w in lower for w in ["enable macros", "open the attachment", "download attachment"])),
                "text_length": len(text),
                "exclamation_count": text.count("!"),
            })
        return rows

def main():
    data, is_demo = load_dataset()
    X = data["text"]
    y = data["label"]

    # Keep a held-out test set. Stratification preserves class proportions.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    features = FeatureUnion([
        ("text", Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, stop_words="english", ngram_range=(1, 2),
                max_features=12000, sublinear_tf=True
            ))
        ])),
        ("url_indicators", Pipeline([
            ("extract", EmailURLFeatures()),
            ("vectorize", DictVectorizer(sparse=True))
        ]))
    ])

    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("\n" + "=" * 58)
    print("PHISHING EMAIL DETECTION RESULTS")
    print("=" * 58)
    print(f"Dataset type: {'DEMONSTRATION (not reliable for real-world accuracy)' if is_demo else 'USER-PROVIDED DATASET'}")
    print(f"Training emails: {len(X_train)} | Test emails: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["phishing", "safe"], zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=["phishing", "safe"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing", "Safe"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=160)
    print(f"Confusion matrix saved to: {OUTPUT_PATH}")

    print("\nTry classifying a new email (type 'quit' to stop).")
    print("Note: this is a machine-learning prediction, not a guarantee of safety.")
    while True:
        try:
            email = input("\nPaste email text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if email.lower() in {"quit", "exit"}:
            break
        if not email:
            print("Please enter some email text.")
            continue
        label = model.predict([email])[0]
        print(f"Prediction: {'PHISHING' if label == 'phishing' else 'SAFE'}")
        if hasattr(model.named_steps["classifier"], "predict_proba"):
            probs = model.predict_proba([email])[0]
            class_order = list(model.named_steps["classifier"].classes_)
            phishing_prob = probs[class_order.index("phishing")]
            print(f"Model phishing score: {phishing_prob:.1%} (not a calibrated security guarantee)")

if __name__ == "__main__":
    main()
