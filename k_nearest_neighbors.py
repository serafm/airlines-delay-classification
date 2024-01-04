import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighborsClassifier:

    def __init__(self, filename):
        self.filename = filename
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.dataset = None
        self.x_airlines = None
        self.y_airlines = None

    def predict(self):
        best_cross_validation_score = (0, 0)
        best_train_test_score = (0, 0)
        best_k = 0
        neighbors = [3, 5, 8, 10]

        for k in neighbors:
            # initialize k-nearest neighbors classifier
            knn_classifier = KNeighborsClassifier(k, weights="distance")

            # fit classifier on training data
            knn_classifier.fit(self.X_train, self.y_train)

            # evaluate classifier on test data
            test_predict = knn_classifier.predict(self.X_test)
            test_acc = accuracy_score(test_predict, self.y_test)
            test_f1 = f1_score(test_predict, self.y_test, average='weighted')
            train_test_score = (test_acc, test_f1)

            # evaluate classifier using cross-validation on training data
            cv_acc_scores = cross_val_score(knn_classifier, self.X_train, self.y_train, cv=10)
            cv_accuracy = cv_acc_scores.mean()

            cv_f1_scores = cross_val_score(knn_classifier, self.X_train, self.y_train, cv=10, scoring='f1_weighted')
            cv_f1score = cv_f1_scores.mean()

            cross_validation_score = (cv_accuracy, cv_f1score)

            if cross_validation_score > best_cross_validation_score and train_test_score > best_train_test_score:
                best_cross_validation_score = cross_validation_score
                best_train_test_score = train_test_score
                best_k = k

        print("Best Score")
        print(best_k, "Nearest Neighbors")
        print("Accuracy:", f'{best_train_test_score[0]:.2f}')
        print("F1 score:", f'{best_train_test_score[1]:.2f}')
        print("Cross Validation Score")
        print("Mean Accuracy:", f'{best_cross_validation_score[0]:.2f}')
        print("Mean F1 score:", f'{best_cross_validation_score[1]:.2f}')

    def read_data(self):
        self.dataset = pd.read_csv(self.filename)
        self.dataset.drop("Flight", axis=1, inplace=True)

    def normalization(self):
        # Creating an instance of label Encoder.
        encode_labels = LabelEncoder()

        # Using .fit_transform function to fit label
        # encoder and return encoded label
        airline_label = encode_labels.fit_transform(self.dataset['Airline'])
        airportFrom_labels = encode_labels.fit_transform(self.dataset['AirportFrom'])
        airportTo_labels = encode_labels.fit_transform(self.dataset['AirportTo'])

        # Appending the array to our dataFrame
        # with column name 'Airline'
        self.dataset["Airline"] = airline_label

        # Appending the array to our dataFrame
        # with column name 'AirportFrom'
        self.dataset["AirportFrom"] = airportFrom_labels

        # Appending the array to our dataFrame
        # with column name 'AirportTo'
        self.dataset["AirportTo"] = airportTo_labels

        self.x_airlines = self.dataset.drop(['Class'], axis=1)
        self.y_airlines = self.dataset['Class']

    # Split data to train and test sets
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_airlines, self.y_airlines,
                                                                                test_size=30,
                                                                                random_state=42)

    def run(self) -> None:
        self.read_data()
        self.normalization()
        self.split_data()
        self.predict()


knn = KNearestNeighborsClassifier("data/airlines_delay.csv")
knn.run()
