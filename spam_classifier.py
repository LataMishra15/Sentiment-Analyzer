import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Training data
messages = [
    ("Congratulations! You won a free lottery ticket", "spam"),
    ("Win a free iPhone now", "spam"),
    ("You have won $1000, claim your prize", "spam"),
    ("Click here to get your free gift", "spam"),

    ("Hey, are we meeting today?", "ham"),
    ("Can you send me the notes?", "ham"),
    ("I will call you later", "ham"),
    ("Don't forget to attend the class tomorrow", "ham")
]

stop_words = set(stopwords.words('english'))

def extract_features(message):
    words = word_tokenize(message.lower())
    words = [word for word in words if word.isalpha()]

    features = {}

    for word in words:
        if word not in stop_words:
            features[word] = True

    return features


# Create training data
training_data = []

for message, label in messages:
    features = extract_features(message)
    training_data.append((features, label))


# Train classifier
classifier = NaiveBayesClassifier.train(training_data)


# Take message from user
message = input("Enter a message: ")

features = extract_features(message)

result = classifier.classify(features)

print("Classification:", result.upper())

if result == "spam":
    print("This message is SPAM.")
else:
    print("This message is NOT SPAM.")