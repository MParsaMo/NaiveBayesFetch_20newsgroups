# 🧠 Text Classification with Naive Bayes on 20 Newsgroups Dataset

This project demonstrates how to build a **text classification pipeline** using **Scikit-learn** to classify documents from the well-known **20 Newsgroups dataset**. The model classifies news articles into categories like `comp.graphics`, `alt.atheism`, `sci.med`, and `soc.religion.christian` using **CountVectorizer**, **TF-IDF**, and a **Multinomial Naive Bayes** classifier.

---

## 📂 Project Structure

- **`load_and_prepare_data()`**: Downloads and loads the dataset (train/test split).
- **`train_text_classifier()`**: Trains the Naive Bayes classifier inside a Scikit-learn pipeline.
- **`evaluate_model()`**: Outputs accuracy, confusion matrix, and classification report.
- **`predict_new_documents()`**: Makes predictions on new, unseen text inputs.

---

## 🔧 Installation

Make sure you have Python ≥ 3.7. Then install the required libraries:

```bash
pip install numpy pandas scikit-learn



