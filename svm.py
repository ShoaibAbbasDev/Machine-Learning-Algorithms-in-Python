# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (emails)
data = [
    ('Free entry in a contest! Click now', 'spam'),
    ('Your loan has been approved', 'spam'),
    ('Meeting at 10 AM tomorrow', 'ham'),
    ('Your Amazon order has been shipped', 'ham'),
    ('Congratulations! You have won a prize', 'spam'),
    ('Reminder for your doctor appointment', 'ham'),
    ('Call me when you get this message', 'ham')
]

# Separate the texts and labels
texts, labels = zip(*data)

# Convert text data into numerical data using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Initialize the SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')

# Train the model
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
