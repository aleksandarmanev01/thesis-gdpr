from sklearn.feature_extraction.text import TfidfVectorizer


def train_and_predict_conventional(model, X_train, y_train, X_test):
    """
    Train a given model and predict on test data.
    """
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return y_pred
