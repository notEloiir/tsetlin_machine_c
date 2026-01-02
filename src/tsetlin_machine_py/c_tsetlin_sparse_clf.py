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


class CSparseTsetlinMachine(ctypes.Structure):
    _fields_ = [
        ("num_classes", ctypes.c_uint32),
        ("threshold", ctypes.c_uint32),
        ("num_literals", ctypes.c_uint32),
        ("num_clauses", ctypes.c_uint32),
        ("max_state", ctypes.c_int8),
        ("min_state", ctypes.c_int8),
        ("sparse_init_state", ctypes.c_int8),
        ("sparse_min_state", ctypes.c_int8),
        ("boost_true_positive_feedback", ctypes.c_uint8),
        ("s", ctypes.c_float),
        ("y_size", ctypes.c_uint32),
        ("y_element_size", ctypes.c_uint32),
        ("y_eq", ctypes.c_void_p),
        ("output_activation", ctypes.c_void_p),
        ("calculate_feedback", ctypes.c_void_p),
        ("mid_state", ctypes.c_int8),
        ("clause_max_size", ctypes.c_int32),
        ("clause_sizes", ctypes.POINTER(ctypes.c_uint32)),
    ]


class CTsetlinSparseClassifier(ClassifierMixin, BaseEstimator):
    """
    A Scikit-learn compatible Sparse Tsetlin Machine classifier with a C backend.

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
        if lib_dir is None:
            base_path = Path(sysconfig.get_paths()["purelib"])
            if sys.platform == "win32":
                self.lib_dir = base_path / "bin"
            else:
                self.lib_dir = base_path / "lib"
        else:
            self.lib_dir = lib_dir

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

        # Define a pointer to the SparseTsetlinMachine struct
        self.tm_p = ctypes.c_void_p

        # stm_create
        self.lib_tm.stm_create.restype = self.tm_p
        self.lib_tm.stm_create.argtypes = [
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

        # stm_free
        self.lib_tm.stm_free.restype = None
        self.lib_tm.stm_free.argtypes = [
            self.tm_p,  # tm
        ]

        # stm_train
        self.lib_tm.stm_train.restype = None
        self.lib_tm.stm_train.argtypes = [
            self.tm_p,  # tm
            ctypes.POINTER(ctypes.c_uint8),  # X
            ctypes.c_void_p,  # y (will be uint32_t* for classification)
            ctypes.c_uint32,  # rows
            ctypes.c_uint32,  # epochs
        ]

        # stm_predict
        self.lib_tm.stm_predict.restype = None
        self.lib_tm.stm_predict.argtypes = [
            self.tm_p,  # tm
            ctypes.POINTER(ctypes.c_uint8),  # X
            ctypes.c_void_p,  # y_pred (will be uint32_t*)
            ctypes.c_uint32,  # rows
        ]

        # stm_save
        self.lib_tm.stm_save.restype = None
        self.lib_tm.stm_save.argtypes = [
            self.tm_p,  # tm
            ctypes.c_char_p,  # filename
        ]

        # stm_load_dense
        self.lib_tm.stm_load_dense.restype = self.tm_p
        self.lib_tm.stm_load_dense.argtypes = [
            ctypes.c_char_p,  # filename
            ctypes.c_uint32,  # y_size
            ctypes.c_uint32,  # y_element_size
        ]

        # stm_save_fbs
        if hasattr(self.lib_tm, "stm_save_fbs"):
            self.lib_tm.stm_save_fbs.restype = None
            self.lib_tm.stm_save_fbs.argtypes = [
                self.tm_p,  # tm
                ctypes.c_char_p,  # filename
            ]

        # stm_load_fbs
        if hasattr(self.lib_tm, "stm_load_fbs"):
            self.lib_tm.stm_load_fbs.restype = self.tm_p
            self.lib_tm.stm_load_fbs.argtypes = [
                ctypes.c_char_p,  # filename
                ctypes.c_uint32,  # y_size
                ctypes.c_uint32,  # y_element_size
            ]

    def __getstate__(self):
        """Prepare the instance for pickling."""
        state = self.__dict__.copy()
        # Remove the un-picklable C-level objects before serialization
        if "lib_tm" in state:
            del state["lib_tm"]
        if "tm_instance_" in state:
            del state["tm_instance_"]
        return state

    def __setstate__(self, state):
        """Restore the instance after unpickling."""
        self.__dict__.update(state)
        # The C library will be loaded on-demand by _load_and_configure_lib()
        # when fit() or predict() is called in the new process.
        self.lib_tm = None
        self.tm_instance_ = None

    def _update_from_c_model(self):
        """Update Python attributes from the loaded C model."""
        if not self.tm_instance_:
            return

        tm_ptr = ctypes.cast(self.tm_instance_, ctypes.POINTER(CSparseTsetlinMachine))
        tm = tm_ptr.contents

        self.n_classes_ = tm.num_classes
        self.threshold = tm.threshold
        self.n_features_in_ = tm.num_literals
        self.num_clauses = tm.num_clauses
        self.max_state = tm.max_state
        self.min_state = tm.min_state
        self.boost_true_positive_feedback = bool(tm.boost_true_positive_feedback)
        self.s = tm.s

        # Restore classes_ and label_encoder_
        # We assume classes are 0..n_classes-1 since we don't save labels in C
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(np.arange(self.n_classes_))
        self.classes_ = self.label_encoder_.classes_

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

        if self.tm_instance_ is not None:
            self.lib_tm.stm_free(self.tm_instance_)

        self.tm_instance_ = self.lib_tm.stm_create(
            self.n_classes_,
            self.threshold,
            self.n_features_in_,
            self.num_clauses,
            self.max_state,
            self.min_state,
            int(self.boost_true_positive_feedback),
            1,  # y_size
            ctypes.sizeof(ctypes.c_uint32),
            self.s,
            self.seed_,
        )
        if not self.tm_instance_:
            raise RuntimeError("Failed to create Sparse Tsetlin Machine in C backend.")

        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_mapped_ptr = y_mapped.ctypes.data_as(ctypes.c_void_p)

        self.lib_tm.stm_train(
            self.tm_instance_,
            X_ptr,
            y_mapped_ptr,
            n_samples,
            self.epochs,
        )

        self.is_fitted_ = True
        return self

    def init_empty_state(self, n_features, classes):
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
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        check_classification_targets(classes)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(classes)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"This classifier needs at least 2 classes; got {self.n_classes_}"
            )

        self.n_features_in_ = int(n_features)

        random_state_instance = check_random_state(self.random_state)
        self.seed_ = random_state_instance.randint(0, 2**32, dtype=np.uint32)

        if self.tm_instance_ is not None:
            self.lib_tm.stm_free(self.tm_instance_)

        self.tm_instance_ = self.lib_tm.stm_create(
            self.n_classes_,
            self.threshold,
            self.n_features_in_,
            self.num_clauses,
            self.max_state,
            self.min_state,
            int(self.boost_true_positive_feedback),
            1,  # y_size
            ctypes.sizeof(ctypes.c_uint32),
            self.s,
            self.seed_,
        )
        if not self.tm_instance_:
            raise RuntimeError("Failed to create Sparse Tsetlin Machine in C backend.")

        # Allow predictions on the untrained model
        self.is_fitted_ = True

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, epochs=None):
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

        if self.tm_instance_ is None:
            full_classes = classes if classes is not None else np.unique(y)
            self.init_empty_state(X.shape[1], full_classes)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature mismatch: {X.shape[1]} != {self.n_features_in_}")
        elif classes is not None and not np.array_equal(classes, self.classes_):
            raise ValueError("Classes inconsistent with previous fit")

        y_mapped = np.ascontiguousarray(
            self.label_encoder_.transform(y),
            dtype=np.uint32,
        )

        n_samples = X.shape[0]
        epochs_to_use = int(self.epochs if epochs is None else epochs)

        X_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        y_mapped_ptr = y_mapped.ctypes.data_as(ctypes.c_void_p)

        self.lib_tm.stm_train(
            self.tm_instance_,
            X_ptr,
            y_mapped_ptr,
            n_samples,
            epochs_to_use,
        )

        self.is_fitted_ = True
        return self

    def reset(self):
        """Free the C Tsetlin Machine and clear the Python-side state."""
        if self.tm_instance_ is not None:
            self._load_and_configure_lib()
            if self.lib_tm is not None:
                self.lib_tm.stm_free(self.tm_instance_)
            self.tm_instance_ = None

        for attr in [
            "is_fitted_",
            "classes_",
            "n_classes_",
            "n_features_in_",
            "label_encoder_",
            "seed_",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

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

        self.lib_tm.stm_predict(self.tm_instance_, X_ptr, y_pred_ptr, n_samples)

        y_pred = self.label_encoder_.inverse_transform(y_pred_mapped)
        return y_pred

    def save_model(self, path):
        """
        Save the current Tsetlin Machine state to a binary file.

        Parameters
        ----------
        path : str or Path
            File path to save the model.
        """
        check_is_fitted(self, "is_fitted_")
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        path_bytes = str(path).encode("utf-8")
        self.lib_tm.stm_save(self.tm_instance_, path_bytes)

    def load_model_dense(self, path):
        """
        Load a Tsetlin Machine state from a dense binary file.
        
        Note: This loads a model saved by the dense Tsetlin Machine implementation.
        It does NOT load a model saved by this sparse implementation's save_model.

        Parameters
        ----------
        path : str or Path
            File path to load the model from.
        """
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        if self.tm_instance_ is not None:
            self.lib_tm.stm_free(self.tm_instance_)

        path_bytes = str(path).encode("utf-8")
        # For classification, y_size is 1 and element size is uint32
        self.tm_instance_ = self.lib_tm.stm_load_dense(
            path_bytes, 1, ctypes.sizeof(ctypes.c_uint32)
        )

        if not self.tm_instance_:
            raise RuntimeError(f"Failed to load model from {path}")

        self._update_from_c_model()
        self.is_fitted_ = True

    def save_model_fbs(self, path):
        """
        Save the current Tsetlin Machine state to a FlatBuffers file.

        Parameters
        ----------
        path : str or Path
            File path to save the model.
        """
        check_is_fitted(self, "is_fitted_")
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        if not hasattr(self.lib_tm, "stm_save_fbs"):
            raise NotImplementedError(
                "The C library was compiled without FlatBuffers support."
            )

        path_bytes = str(path).encode("utf-8")
        self.lib_tm.stm_save_fbs(self.tm_instance_, path_bytes)

    def load_model_fbs(self, path):
        """
        Load a Tsetlin Machine state from a FlatBuffers file.

        Parameters
        ----------
        path : str or Path
            File path to load the model from.
        """
        self._load_and_configure_lib()
        if self.lib_tm is None:
            raise RuntimeError("C library not loaded.")

        if not hasattr(self.lib_tm, "stm_load_fbs"):
            raise NotImplementedError(
                "The C library was compiled without FlatBuffers support."
            )

        if self.tm_instance_ is not None:
            self.lib_tm.stm_free(self.tm_instance_)

        path_bytes = str(path).encode("utf-8")
        # For classification, y_size is 1 and element size is uint32
        self.tm_instance_ = self.lib_tm.stm_load_fbs(
            path_bytes, 1, ctypes.sizeof(ctypes.c_uint32)
        )

        if not self.tm_instance_:
            raise RuntimeError(f"Failed to load model from {path}")

        self._update_from_c_model()
        self.is_fitted_ = True

    def __del__(self):
        if hasattr(self, "tm_instance_") and self.tm_instance_ is not None:
            if not hasattr(self, "lib_tm") or self.lib_tm is None:
                self._load_and_configure_lib()
            if self.lib_tm is None:
                raise RuntimeError("C library not loaded.")
            self.lib_tm.stm_free(self.tm_instance_)

    def _more_tags(self):
        return {"binary_only": True}

    def estimate_model_size(self):
        """
        Estimate the model size in bytes based on the C implementation's memory allocation.

        Returns
        -------
        int
            Estimated model size in bytes.
        """
        check_is_fitted(self)

        if self.tm_instance_ is None:
            raise RuntimeError("Model is fitted but C instance is None.")

        tm_ptr = ctypes.cast(self.tm_instance_, ctypes.POINTER(CSparseTsetlinMachine))
        tm = tm_ptr.contents
        
        # Read clause_sizes array
        clause_sizes_arr = ctypes.cast(tm.clause_sizes, ctypes.POINTER(ctypes.c_uint32 * self.num_clauses)).contents
        total_literals = sum(clause_sizes_arr)
        
        # Size of Tsetlin Automaton states (linked list nodes)
        # struct TAStateNode { uint32_t ta_id; int8_t ta_state; struct TAStateNode *next; };
        # Approx 16 bytes per node
        ta_state_bytes = total_literals * 16
        
        # Pointers to linked lists (ta_state)
        ta_state_ptrs_bytes = self.num_clauses * ctypes.sizeof(ctypes.c_void_p)
        
        # Pointers to active literals (active_literals)
        active_literals_ptrs_bytes = self.n_classes_ * ctypes.sizeof(ctypes.c_void_p)
        
        # Clause sizes array
        clause_sizes_bytes = self.num_clauses * ctypes.sizeof(ctypes.c_uint32)

        # Size of clause weights
        weights_bytes = (
            self.num_clauses * self.n_classes_ * ctypes.sizeof(ctypes.c_int16)
        )

        # Size of the clause output buffer
        clause_output_bytes = self.num_clauses * ctypes.sizeof(ctypes.c_uint8)

        # Size of the votes buffer
        votes_bytes = self.n_classes_ * ctypes.sizeof(ctypes.c_int32)

        total_bytes = (
            ta_state_bytes
            + ta_state_ptrs_bytes
            + active_literals_ptrs_bytes
            + clause_sizes_bytes
            + weights_bytes
            + clause_output_bytes
            + votes_bytes
        )

        return total_bytes
