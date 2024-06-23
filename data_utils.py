import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    df_fraud = data.loc[data.fraud == 1]
    df_non_fraud = data.loc[data.fraud == 0]

    data_reduced = data.drop(['step','zipcodeOri', 'zipMerchant'], axis=1)
    
    col_categorical = data_reduced.select_dtypes(include=['object']).columns
    mappings = {}
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')
        mappings[col] = dict(enumerate(data_reduced[col].cat.categories))
        data_reduced[col] = data_reduced[col].cat.codes

    X = data_reduced.drop(['fraud'], axis=1)
    y = data['fraud']
    
    return X, y, df_fraud, df_non_fraud, mappings

def plot_count(data):
    sns.set()
    sns.countplot(x="fraud", data=data)
    plt.title("Count of Fraudulent Payments")
    plt.show()

def plot_histogram(df_fraud, df_non_fraud):
    plt.hist(df_fraud.amount, alpha=0.5, label='fraud', bins=100)
    plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud', bins=100)
    plt.title("Histogram for fraud and nonfraud payments")
    plt.ylim(0, 10000)
    plt.xlim(0, 1000)
    plt.legend()
    plt.show()

def plot_roc_auc(y_test, preds):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
