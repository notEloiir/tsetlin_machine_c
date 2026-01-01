from numbers import Integral, Real

import green_tsetlin as gt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class GreenTsetlinClassifier(ClassifierMixin, BaseEstimator):
    """
    A Scikit-learn compatible Green Tsetlin Machine classifier.

    Parameters
    ----------
    n_clauses : int, default=1000
        The number of clauses in the Tsetlin Machine.
    s : float, default=3.0
        The S hyperparameter for the Tsetlin Machine.
    threshold : int, default=1000
        The T threshold for the Tsetlin Machine.
    literal_budget : int, default=None
        The literal budget for the Tsetlin Machine.
    boost_true_positives : bool, default=False
        Whether to boost true positive feedback.
    n_epochs : int, default=10
        The number of epochs to train for.
    random_state : int, default=None
        Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        Number of parallel jobs to use for training.
        -1 means using all processors.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        > 0 enables progress bar.
    """

    _parameter_constraints: dict = {
        "n_clauses": [Interval(Integral, 1, None, closed="left")],
        "s": [Interval(Real, 1, None, closed="neither")],
        "threshold": [Interval(Integral, 1, None, closed="left")],
        "literal_budget": [Interval(Integral, 1, None, closed="left"), None],
        "boost_true_positives": [bool],
        "n_epochs": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
        "verbose": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_clauses=1000,
        s=3.0,
        threshold=1000,
        literal_budget=None,
        boost_true_positives=False,
        n_epochs=10,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        self.n_clauses = n_clauses
        self.s = s
        self.threshold = threshold
        self.literal_budget = literal_budget
        self.boost_true_positives = boost_true_positives
        self.n_epochs = n_epochs
        if random_state is None:
            random_state = np.random.randint(0, 2**31 - 1)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the Green Tsetlin Machine classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Must be binary (0 or 1).
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.uint8)  # Ensure X is uint8 as per examples
        check_classification_targets(y)
        if np.any((X != 0) & (X != 1)):  # Ensure X is binary (0 or 1)
            raise ValueError("Input X must be binary (contain only 0s and 1s).")

        self.label_encoder_ = LabelEncoder()
        y_mapped = self.label_encoder_.fit_transform(y).astype(np.uint32)
        self.classes_ = self.label_encoder_.classes_

        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError(
                "This classifier needs at least 2 classes; got %d class"
                % self.n_classes_
            )

        self.n_features_in_ = X.shape[1]

        self.tm_ = gt.TsetlinMachine(
            n_literals=self.n_features_in_,
            n_clauses=self.n_clauses,
            n_classes=self.n_classes_,
            s=float(self.s),
            threshold=self.threshold,
            literal_budget=self.literal_budget,
            boost_true_positives=self.boost_true_positives,
        )

        trainer = gt.Trainer(
            self.tm_,
            seed=self.random_state,
            n_jobs=self.n_jobs,
            n_epochs=self.n_epochs,
            progress_bar=(self.verbose > 0),
        )

        # Green Tsetlin expects X to be uint8 and y to be uint32
        # X is already checked for uint8. y_mapped is uint32.
        trainer.set_train_data(X, y_mapped)
        # Using training data as eval data as per the example,
        # this is often for monitoring training progress.
        trainer.set_eval_data(X, y_mapped)
        trainer.train()

        self.predictor_ = self.tm_.get_predictor()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Must be binary (0 or 1).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.uint8)  # Ensure X is uint8
        if np.any((X != 0) & (X != 1)):  # Ensure X is binary (0 or 1)
            raise ValueError("Input X must be binary (contain only 0s and 1s).")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features of the input must be {self.n_features_in_}, got {X.shape[1]}"
            )

        # The gt.Predictor.predict method expects a single sample.
        # We can vectorize it or loop. Vectorizing is generally cleaner.
        # However, to closely match the example's loop:
        # y_pred_mapped = np.array([self.predictor_.predict(x) for x in X])

        # Using np.vectorize as seen in other green_tsetlin examples for efficiency
        vectorized_predict = np.vectorize(self.predictor_.predict, signature="(n)->()")
        y_pred_mapped = vectorized_predict(X)

        # Map predictions back to original class labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_mapped)

        return y_pred

    def _more_tags(self):
        # Indicate that the input X must be binary
        return {"binary_only": True}

    def estimate_model_size(self):
        """
        Estimate the model size in bytes based on a specified internal structure.

        The estimation includes memory for:
        - Clauses: (n_clauses, n_literals*2) of np.int8
        - Clause weights: (n_clauses, n_classes) of np.int8
        - Class votes: (n_classes) of np.int32
        - Literal counts: (n_clauses) of np.int32

        Returns
        -------
        int
            Estimated model size in bytes.
        """
        check_is_fitted(self)

        # n_literals corresponds to self.n_features_in_
        n_literals = self.n_features_in_

        # Calculate size for clauses: (self.n_clauses, n_literals*2) of np.int8
        clauses_bytes = self.n_clauses * (n_literals * 2) * np.dtype(np.int8).itemsize

        # Calculate size for clause_weights: (self.n_clauses, self.n_classes_) of np.int8
        clause_weights_bytes = self.n_clauses * self.n_classes_ * np.dtype(np.int8).itemsize

        # Calculate size for class_votes: (self.n_classes_) of np.int32
        class_votes_bytes = self.n_classes_ * np.dtype(np.int32).itemsize

        # Calculate size for literal_counts: (self.n_clauses) of np.int32
        literal_counts_bytes = self.n_clauses * np.dtype(np.int32).itemsize

        total_bytes = clauses_bytes + clause_weights_bytes + class_votes_bytes + literal_counts_bytes
        return total_bytes
