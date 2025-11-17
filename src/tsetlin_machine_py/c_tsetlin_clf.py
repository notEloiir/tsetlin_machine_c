import ctypes
import sys
import sysconfig
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    _fit_context,  # type: ignore[attr-defined]
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CTsetlinClassifier(ClassifierMixin, BaseEstimator):
    """
    A Scikit-learn compatible Tsetlin Machine classifier with a C backend.

    Parameters
    ----------
    threshold : int, default=1000
        The T threshold for the Tsetlin Machine.
    num_clauses : int, default=1000
        The number of clauses in the Tsetlin Machine.
    max_state : int, default=127
        The maximum state for the Tsetlin Machine.
    min_state : int, default=-127
        The minimum state for the Tsetlin Machine.
    boost_true_positive_feedback : bool, default=False
        Whether to boost true positive feedback.
    s : float, default=3.0
        The learning sensitivity parameter for the Tsetlin Machine.
    epochs : int, default=10
        Number of training epochs.
    random_state : int, default=None
        Controls the randomness of the estimator.
    lib_dir : str, default=None
        Path to the compiled C libraries directory.
    """

    _parameter_constraints: dict = {
        "threshold": [Interval(Integral, 1, None, closed="left")],  # type: ignore[reportAbstractUsage]
        "num_clauses": [Interval(Integral, 1, None, closed="left")],  # type: ignore[reportAbstractUsage]
        "max_state": [Interval(Integral, -128, 127, closed="both")],  # type: ignore[reportAbstractUsage]
        "min_state": [Interval(Integral, -128, 127, closed="both")],  # type: ignore[reportAbstractUsage]
        "boost_true_positive_feedback": [bool],
        "s": [Interval(Real, 1.0, None, closed="left")],  # type: ignore[reportAbstractUsage]
        "epochs": [Interval(Integral, 1, None, closed="left")],  # type: ignore[reportAbstractUsage]
        "random_state": ["random_state", None],
        "lib_dir": [str, Path, None],
    }

    def __init__(
        self,
        threshold=1000,
        num_clauses=1000,
        max_state=127,
        min_state=-127,
        boost_true_positive_feedback=False,
        s=3.0,
        epochs=10,
        random_state=None,
        lib_dir=None,
    ):
        self.threshold = threshold
        self.num_clauses = num_clauses
        self.max_state = max_state
        self.min_state = min_state
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.s = s
        self.epochs = epochs
        self.random_state = random_state
        self.lib_dir = (
            (Path(sysconfig.get_paths()["purelib"]) / "lib")
            if lib_dir is None
            else lib_dir
        )

        self.lib_tm = None
        self.tm_instance_ = None

    def _load_and_configure_lib(self):
        """
        Helper function to lazy-load the C library and configure functions
        in a cross-platform way.
        """
        # Only load if it hasn't been loaded yet.
        if self.lib_tm is not None:
            return

        if sys.platform == "win32":  # windows
            flatcc_name = "libflatccrt.dll"
            tsetlin_name = "libtsetlin_machine_c.dll"
            load_mode = 0
        elif sys.platform == "darwin":  # macOS
            flatcc_name = "libflatccrt.dylib"
            tsetlin_name = "libtsetlin_machine_c.dylib"
            load_mode = ctypes.RTLD_GLOBAL
        else:  # linux and other POSIX
            flatcc_name = "libflatccrt.so"
            tsetlin_name = "libtsetlin_machine_c.so"
            load_mode = ctypes.RTLD_GLOBAL

        flatcc_path = self.lib_dir / flatcc_name
        tsetlin_path = self.lib_dir / tsetlin_name

        try:
            if flatcc_path.exists():
                ctypes.CDLL(flatcc_path.resolve(), mode=load_mode)
            self.lib_tm = ctypes.CDLL(tsetlin_path.resolve(), mode=load_mode)

            self._configure_c_functions()
        except OSError as e:
            raise OSError(
                f"Could not load libraries from {self.lib_dir}. "
                f"Ensure it's compiled and path is correct. Error: {e}"
            )

    def _configure_c_functions(self):
        """Define argument types and return types for C functions."""
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        # Define a pointer to the TsetlinMachine struct
        self.tm_p = ctypes.c_void_p

        # tm_create
        self.lib_tm.tm_create.restype = self.tm_p
        self.lib_tm.tm_create.argtypes = [
            ctypes.c_uint32,  # num_classes
            ctypes.c_uint32,  # threshold
            ctypes.c_uint32,  # num_literals
            ctypes.c_uint32,  # num clauses
            ctypes.c_int8,  # max_state
            ctypes.c_int8,  # min_state
            ctypes.c_uint8,  # boost_true_positive_feedback
            ctypes.c_uint32,  # y_size
            ctypes.c_uint32,  # y_element_size
            ctypes.c_float,  # learning sensitivity (s)
            ctypes.c_uint32,  # seed
        ]

        # tm_free
        self.lib_tm.tm_free.restype = None
        self.lib_tm.tm_free.argtypes = [
            self.tm_p,  # tm
        ]

        # tm_train
        self.lib_tm.tm_train.restype = None
        self.lib_tm.tm_train.argtypes = [
            self.tm_p,  # tm
            ctypes.POINTER(ctypes.c_uint8),  # X
            ctypes.c_void_p,  # y (will be uint32_t* for classification)
            ctypes.c_uint32,  # rows
            ctypes.c_uint32,  # epochs
        ]

        # tm_predict
        self.lib_tm.tm_predict.restype = None
        self.lib_tm.tm_predict.argtypes = [
            self.tm_p,  # tm
            ctypes.POINTER(ctypes.c_uint8),  # X
            ctypes.c_void_p,  # y_pred (will be uint32_t*)
            ctypes.c_uint32,  # rows
        ]

    def __getstate__(self):
        """Prepare the instance for pickling."""
        state = self.__dict__.copy()
        # Remove the un-picklable C-level objects before serialization
        if "lib_tm" in state:
            del state["lib_tm"]
        if "_tm_instance" in state:
            del state["_tm_instance"]
        return state

    def __setstate__(self, state):
        """Restore the instance after unpickling."""
        self.__dict__.update(state)
        # The C library will be loaded on-demand by _load_and_configure_lib()
        # when fit() or predict() is called in the new process.
        self.lib_tm = None
        self._tm_instance = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the Tsetlin Machine classifier.

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
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        X, y = check_X_y(X, y, dtype=np.uint8)
        check_classification_targets(y)
        if np.any((X != 0) & (X != 1)):
            raise ValueError("Input X must be binary (contain only 0s and 1s).")

        self.label_encoder_ = LabelEncoder()
        y_mapped = np.ascontiguousarray(
            self.label_encoder_.fit_transform(y),
            dtype=np.uint32,
        )
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"This classifier needs at least 2 classes; got {self.n_classes_}"
            )

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        random_state_instance = check_random_state(self.random_state)
        self.seed_ = random_state_instance.randint(0, 2**32, dtype=np.uint32)

        self.tm_instance_ = self.lib_tm.tm_create(
            self.n_classes_,
            self.threshold,
            self.n_features_in_,
            self.num_clauses,
            self.max_state,
            self.min_state,
            int(self.boost_true_positive_feedback),
            1,
            ctypes.sizeof(ctypes.c_uint32),
            self.s,
            self.seed_,
        )
        if not self.tm_instance_:
            raise RuntimeError("Failed to create Tsetlin Machine in C backend.")

        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_mapped_ptr = y_mapped.ctypes.data_as(ctypes.c_void_p)

        self.lib_tm.tm_train(
            self.tm_instance_,
            X_ptr,
            y_mapped_ptr,
            n_samples,
            self.epochs,
        )

        self.is_fitted_ = True
        return self

    def init_empty_state(self, n_features, classes):  # TODO: needs testing
        """
        Allocate the C Tsetlin Machine once without training.

        Parameters
        ----------
        n_features : int
            Number of binary features (number of literals expected).
        classes : array-like
            Full list of classes the model will handle.

        Notes
        -----
        After calling this method you can call predict(). The model
        will produce outputs based on its untrained (initial) state.
        Use partial_fit or fit to train for meaningful predictions.
        """
        # Load C lib
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        # Fit label encoder on the full class set
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(classes)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"This classifier needs at least 2 classes; got {self.n_classes_}"
            )

        # Store feature count
        self.n_features_in_ = int(n_features)

        # Seed
        random_state_instance = check_random_state(self.random_state)
        self.seed_ = random_state_instance.randint(0, 2**32, dtype=np.uint32)

        # Create empty TM instance
        self.tm_instance_ = self.lib_tm.tm_create(
            self.n_classes_,
            self.threshold,
            self.n_features_in_,
            self.num_clauses,
            self.max_state,
            self.min_state,
            int(self.boost_true_positive_feedback),
            1,
            ctypes.sizeof(ctypes.c_uint32),
            self.s,
            self.seed_,
        )
        if not self.tm_instance_:
            raise RuntimeError("Failed to create Tsetlin Machine in C backend.")

        # Mark as fitted for prediction availability (untrained initial state).
        self.is_fitted_ = True

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, epochs=None):  # TODO: needs testing
        """
        Incrementally train the existing TM state. First call will initialize
        an empty state if needed (using `classes` or the unique labels in `y`).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Binary features.
        y : array-like, shape (n_samples,)
            Targets with labels contained in `classes`.
        classes : array-like, optional
            Full set of classes. Required on first call if state not initialized.
        epochs : int, optional
            Epochs to train this call. Defaults to self.epochs.

        Returns
        -------
        self
        """
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        X, y = check_X_y(X, y, dtype=np.uint8)
        check_classification_targets(y)
        if np.any((X != 0) & (X != 1)):
            raise ValueError("Input X must be binary (contain only 0s and 1s).")

        if self.tm_instance_ is None:
            full_classes = classes if classes is not None else np.unique(y)
            self.init_empty_state(X.shape[1], full_classes)
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Number of features of the input must be {self.n_features_in_}, got {X.shape[1]}"
                )
            if classes is not None:
                provided = np.array(classes)
                if provided.shape != self.classes_.shape or np.any(
                    provided != self.classes_
                ):
                    raise ValueError(
                        "Provided classes do not match the classes seen during initialization."
                    )

        # Map y to encoded ints
        y_mapped = np.ascontiguousarray(
            self.label_encoder_.transform(y),
            dtype=np.uint32,
        )

        # Train incrementally
        n_samples = X.shape[0]
        epochs_to_use = int(self.epochs if epochs is None else epochs)

        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_mapped_ptr = y_mapped.ctypes.data_as(ctypes.c_void_p)

        self.lib_tm.tm_train(
            self.tm_instance_,
            X_ptr,
            y_mapped_ptr,
            n_samples,
            epochs_to_use,
        )

        self.is_fitted_ = True
        return self

    def reset(self):  # TODO: needs testing
        """Free the C Tsetlin Machine and clear the Python-side state."""

        # Free C-side instance
        if self.tm_instance_ is not None:
            if self.lib_tm is None:
                # Load lib just to free the memory
                self._load_and_configure_lib()

            if self.lib_tm is not None:
                try:
                    self.lib_tm.tm_free(self.tm_instance_)
                except Exception as e:
                    # Log or warn if freeing fails, but continue
                    print(f"Warning: Failed to free C-side TM instance: {e}")

            self.tm_instance_ = None

        # Clear Python-side learned attributes
        attrs_to_clear = [
            "label_encoder_",
            "classes_",
            "n_classes_",
            "n_features_in_",
            "seed_",
            "is_fitted_",
        ]

        for attr in attrs_to_clear:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Must be binary (0 or 1).

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, "is_fitted_")

        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        X = check_array(X, dtype=np.uint8)
        if np.any((X != 0) & (X != 1)):
            raise ValueError("Input X must be binary (contain only 0s and 1s).")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features of the input must be {self.n_features_in_}, got {X.shape[1]}"
            )

        n_samples = X.shape[0]
        y_pred_mapped = np.zeros(n_samples, dtype=np.uint32)

        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_pred_ptr = y_pred_mapped.ctypes.data_as(ctypes.c_void_p)

        self.lib_tm.tm_predict(self.tm_instance_, X_ptr, y_pred_ptr, n_samples)

        y_pred = self.label_encoder_.inverse_transform(y_pred_mapped)
        return y_pred

    def __del__(self):
        if hasattr(self, "_tm_instance") and self._tm_instance:
            if not hasattr(self, "lib_tm") or self.lib_tm is None:
                self._load_and_configure_lib()
            if self.lib_tm is None:
                raise RuntimeError("C library not loaded.")
            self.lib_tm.tm_free(self._tm_instance)

    def _more_tags(self):
        return {"binary_only": True}

    def estimate_model_size(self):
        """
        Estimate the model size in bytes based on the C implementation's memory allocation.

        The estimation includes memory for:
        - Tsetlin Automaton states: (num_clauses, num_literals, 2) of int8_t
        - Clause weights: (num_clauses, num_classes) of int16_t
        - Clause outputs: (num_clauses) of uint8_t
        - Feedback buffer: (num_clauses, num_classes, 3) of int8_t
        - Class votes: (num_classes) of int32_t

        Returns
        -------
        int
            Estimated model size in bytes.
        """
        check_is_fitted(self)

        # n_literals corresponds to self.n_features_in_
        n_literals = self.n_features_in_

        # Size of Tsetlin Automaton states
        # C code: tm->ta_state = (int8_t *)malloc(num_clauses * num_literals * 2 * sizeof(int8_t));
        ta_state_bytes = (
            self.num_clauses * n_literals * 2 * ctypes.sizeof(ctypes.c_int8)
        )

        # Size of clause weights
        # C code: tm->weights = (int16_t *)malloc(num_clauses * num_classes * sizeof(int16_t));
        weights_bytes = (
            self.num_clauses * self.n_classes_ * ctypes.sizeof(ctypes.c_int16)
        )

        # Size of the clause output buffer
        # C code: tm->clause_output = (uint8_t *)malloc(num_clauses * sizeof(uint8_t));
        clause_output_bytes = self.num_clauses * ctypes.sizeof(ctypes.c_uint8)

        # Size of the feedback buffer
        # C code: tm->feedback = (int8_t *)malloc(num_clauses * num_classes * 3 * sizeof(int8_t));
        feedback_bytes = (
            self.num_clauses * self.n_classes_ * 3 * ctypes.sizeof(ctypes.c_int8)
        )

        # Size of the votes buffer
        # C code: tm->votes = (int32_t *)malloc(num_classes * sizeof(int32_t));
        votes_bytes = self.n_classes_ * ctypes.sizeof(ctypes.c_int32)

        total_bytes = (
            ta_state_bytes
            + weights_bytes
            + clause_output_bytes
            + feedback_bytes
            + votes_bytes
        )

        return total_bytes
