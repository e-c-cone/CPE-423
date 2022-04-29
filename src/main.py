import argparse
import json
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

from processing.election_data import data_generator
from models.model import Model


def arguments():
    parser = argparse.ArgumentParser(description="BEEGUS")

    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Include warning data")
    parser.add_argument("-rd", "--reload_data", default=False, action="store_true",
                        help="Reload data instead of using data saved on file")
    parser.add_argument("-ex", "--exclude", action='store', help="Excludes a model")
    parser.add_argument("-cy", "--cutoff_year", action='store', default=2010, type=int,
                        help="Sets end year for model training")
    parser.add_argument("-t", "--train", action='store_true', default=False,
                        help="Indicates whether to train a model from scratch or to use a pretrained version")
    parser.add_argument("-ft", "--full_train", action='store_true', default=False,
                        help="Generates dataset and trains model 2")

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    verbose = args.verbose

    ###  Generate Election Data  ###
    # Here we clean the data, and reshape it using pandas dataframes to be an appropriate shape for input into the
    # Linear Regression algorithm. We start by generating the possible outputs (percentage of votes for each party)
    # and continue by adding the GDP and Income data. This is not expected to be very accurate.

    # utils.generate_ids_from_cand_dir()
    
    # x, y, additional_data, keys = dg.generate_dataset()
    if args.full_train:
        dg = data_generator(args.cutoff_year, verbose, args.reload_data)

        predicted_results = []
        x_test = None
        y_test = None
        for year in range(1992, args.cutoff_year+1)[::2]:
            x, y, additional_data, keys = dg.generate_dataset()
            train_inds = [(int(key.split('_')[1]) < int(args.cutoff_year)) for key in keys]
            test_inds = [(int(key.split('_')[1]) == int(args.cutoff_year)) for key in keys]

            x = np.array(x)
            y = np.array(y)

            # additional_data.to_csv('TEST.csv')
            x_train = x[train_inds]
            y_train = y[train_inds]
            x_test = x[test_inds]
            y_test = y[test_inds]
            keys = np.array(keys)[test_inds]
            additional_data = pd.DataFrame(additional_data).set_index('election_id').iloc[test_inds]

            if args.exclude:
                model = Model(inp_shape=np.array(x_train).shape[1:], out_shape=np.array(y_train).shape[1],
                              exclude=[args.exclude], verbose=verbose)
            else:
                model = Model(inp_shape=np.array(x_train).shape[1:], out_shape=np.array(y_train).shape[1], verbose=verbose)

            if args.train:
                model.fit(x_train, y_train, plot=(year == args.cutoff_year))
            else:
                try:
                    model.load()
                except FileNotFoundError:
                    logger.warning(f'Model files not found, training model as usual')
                    model.fit(x_train, y_train, plot=(year == args.cutoff_year))

            dataset = dg.election_squish()
            predicted_results += [dataset]
            # print(predicted_results)
            logger.success(f'{year=} completed')
        if args.train:
            model.save()
            model.predict_last_year(x_test, y_test, additional_data, keys)

        dataset = pd.concat(predicted_results, ignore_index=True)
        # print(dataset)
        with open('FULL_DATASET.json', 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, indent=4)
        # dataset.to_csv('FULL_DATASET.csv')

    with open('FULL_DATASET.json') as f:
        dataset = pd.DataFrame(dict(json.load(f)))

    X = dataset['X']
    Y = dataset['Y_BC']
    X = np.array([np.array(x) for x in X])
    Y = np.array([np.array(y) for y in Y])
    print(f'{X.shape=}, {Y.shape=}')

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    def test_classifier(percent_classifier, all_predictions):
        percent_classifier.fit(x_train, y_train)
        predictions = percent_classifier.predict(x_test)
        all_predictions += [predictions]
        TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
        print('True Positive(TP)  = ', TP)
        print('False Positive(FP) = ', FP)
        print('True Negative(TN)  = ', TN)
        print('False Negative(FN) = ', FN)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))

    L = []
    logger.info(f'LOGISTIC REGRESSION')
    pc = LogisticRegression()
    test_classifier(pc, L)

    logger.info(f'RANDOM FOREST CLASSIFIER')
    pc = RandomForestClassifier()
    test_classifier(pc, L)

    logger.info(f'LINEAR SVC')
    pc = LinearSVC()
    test_classifier(pc, L)

    # logger.info(f'GAUSSIAN NB')
    # pc = GaussianNB()
    # test_classifier(pc, L)

    logger.info(f'AGGREGATE')
    L = np.array(L).T
    num_models = np.array(L).shape[1]
    all_preds = np.sum(np.array(L), axis=1)/num_models
    all_preds = [1 if i > 0.5 else 0 for i in all_preds]

    TN, FP, FN, TP = confusion_matrix(y_test, all_preds).ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
