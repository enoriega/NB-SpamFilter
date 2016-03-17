''' Main script '''

import glob, os
import sklearn as sk
import numpy as np
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from message import Message

DATA_DIR='data/data'
LABELS_PATH='data/partial/index'

# Read messages
print "Reading messages ..."
start = datetime.now()
messages = []
for path in glob.glob(os.path.join(DATA_DIR, '*'))[:1000]:
    m = Message.load(path)
    messages.append(m)

# Read the labels
print "Reading labels ..."
with open(LABELS_PATH) as f:
    labels_dict = {}
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels_dict[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0
end = datetime.now()

print "%i seconds reading and parsing ..." % (end - start).seconds

print "Creating feature vectors ..."
start = datetime.now()
# Split the data in 70% training, 30% testing
vectorizer = sk.feature_extraction.text.CountVectorizer()
X = vectorizer.fit_transform([m.processedText for m in messages])
y = np.array([labels_dict[m.id] for m in messages])
end = datetime.now()

print "%i seconds extracting features ..." % (end - start).seconds

# Split the data set in training 70% and testing 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)

print "Training ..."
start = datetime.now()
nb = MultinomialNB()
nb.fit(X_train, y_train)
end = datetime.now()

print "%i seconds fitting the model ..." % (end - start).seconds

print "Testing:"
y_pred = nb.predict(X_test)
print classification_report(y_test, y_pred, target_names=['Spam', 'Ham'])
print
print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)
print
print "Confusion matrix:"
print confusion_matrix(y_test, y_pred)
