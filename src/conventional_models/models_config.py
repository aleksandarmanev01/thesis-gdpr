from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Model definitions
models = {
    'LR': LogisticRegression(class_weight='balanced'),
    'SVM': SVC(class_weight='balanced', gamma='scale', kernel='rbf')
}
