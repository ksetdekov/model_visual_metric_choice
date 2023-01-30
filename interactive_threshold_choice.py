import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

class PrecisionRecall:
    def __init__(self, model, X, y_true):
        self.model = model
        self.X = X
        self.y_true = y_true
        self.y_score = self.model.predict_proba(X)[:, 1]
        self.precision, self.recall, _ = precision_recall_curve(y_true, self.y_score)
        self.avg_precision = average_precision_score(y_true, self.y_score)

    def show_tradeoff(self, threshold):
        y_pred = self.model.predict(self.X)
        y_pred[self.y_score < threshold] = 0
        y_pred[self.y_score >= threshold] = 1
        st.write("Threshold:", threshold)
        st.write("Average Precision:", self.avg_precision)
        # st.line_chart(self.recall, self.precision)
        return y_pred

    def show_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_true, y_pred)
        st.write("Confusion Matrix:", cm)
        st.write("True Positives:", cm[1, 1])
        st.write("True Negatives:", cm[0, 0])
        st.write("False Positives:", cm[0, 1])
        st.write("False Negatives:", cm[1, 0])
    
    def optimize_threshold(self, n_options=4, n_steps=10):
        threshold = st.slider("Threshold", 0, 1.0, value=0.5)
        y = self.y_test
        
        for step in range(n_steps):
            thresholds = [np.random.uniform(threshold - 0.1, threshold + 0.1) for i in range(n_options)]
            y_preds = [self.show_tradeoff(threshold) for threshold in thresholds]
            precisions = [precision_score(y, y_pred) for y_pred in y_preds]
            recalls = [recall_score(y, y_pred) for y_pred in y_preds]

            st.write("Options:")
            for i in range(n_options):
                st.write("Threshold: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(thresholds[i], precisions[i], recalls[i]))
                st.image(confusion_matrix(y, y_preds[i]), use_column_width=True)

            # Select the best threshold
            best_index = st.selectbox("Select the best option", options=range(n_options))
            threshold = thresholds[best_index]

        return threshold

    

# Load the breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data["data"], columns=data["feature_names"])
df["target"] = data["target"]

# Truncate the dataset to 10,000 rows
df = df.iloc[:10000, :]

st.title("Precision-Recall Tradeoff")

X = df.drop("target", axis=1)
y = df["target"]

# Fit the logistic regression model
model = LogisticRegression(random_state=0).fit(X, y)

pr = PrecisionRecall(model, X, y)

# Get the best threshold for precision
precision_threshold = np.max(pr.precision[pr.recall >= 0.7])
y_pred_precision = pr.show_tradeoff(precision_threshold)
st.write("Precision Threshold:", precision_threshold)
st.write('High Precision')
st.table(confusion_matrix(y, y_pred_precision))

# Get the best threshold for recall
recall_threshold = np.max(pr.recall[pr.precision >= 0.7])
y_pred_recall = pr.show_tradeoff(recall_threshold)

st.write("Recall Threshold:", recall_threshold)
st.write('High Recall')
st.table(confusion_matrix(y, y_pred_recall))
# Get the best threshold for precision-recall 
balance_threshold = (precision_threshold + recall_threshold) / 2
y_pred_balance = pr.show_tradeoff(balance_threshold)

st.write("Balance Threshold:", balance_threshold)
st.write('Balanced Precision and Recall')
st.table(confusion_matrix(y, y_pred_balance))
# Show the confusion matrix for the threshold
threshold = st.slider("Threshold", 0.0, 1.0, value=float(balance_threshold))
y_pred = pr.show_tradeoff(threshold)
pr.show_confusion_matrix(y_pred)

