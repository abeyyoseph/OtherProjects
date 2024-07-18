
import pandas as pd 

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix 


def print_confusion_matrix(true_labels, pred_labels):
    (
        _00, _01, 
        _10, _11
    ) = confusion_matrix(true_labels, pred_labels).ravel()
    print('\t\tPred')
    print('\t\tP0\tP1')
    print('True\tT0\t{}\t{}'.format(_00, _01))
    print('\tT1\t{}\t{}'.format(_10, _11))
    print()

    precision = _00 / (_00 + _10) if _00 + _10 else 0.0
    recall = _00 / (_00 + _01)   if _00 + _01 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if _00 > 0 else 0.0
    print('Precision: {:.2f}'.format(precision))
    print('Recall: {:.2f}'.format(recall))
    print('F1: {:.2f}'.format(f1))


def main():
    dataset = pd.read_csv(open('mpii_dataset.csv', 'r'), index_col=None)
    X, y = dataset.drop(columns=['sport']), dataset['sport']

    model = GaussianNB()
    model.fit(X, y)

    y_pred = model.predict(X)
    print_confusion_matrix(y, y_pred)


if __name__ == '__main__':
    main()