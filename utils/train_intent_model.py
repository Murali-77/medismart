import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# Load multilingual model
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load JSONL data
intent_data_path = "../data/intent_data.jsonl"
texts, labels = [], []

with open(intent_data_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        texts.append(entry["text"])
        labels.append(entry["intent"])

# Encode sentences into embeddings
X = embed_model.encode(texts, show_progress_bar=True)

# Encode intent labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train classifier (X = the features the model loads from, y = target values the model tries to predict) (LR classifier - good for categorical, classification tasks; uses sigmoid curve)
# max_iter=300 specifies the maximum number of iterations the optimization algorithm will run during training.
clf = LogisticRegression(max_iter=300)
clf.fit(X, y)

# Save classifier and label encoder
joblib.dump((clf, label_encoder), "../data/intent_clf_transformer.joblib")

# Save embedding model
embed_model.save("../data/embed_model")

print("✅ Multilingual intent classifier trained and saved.")