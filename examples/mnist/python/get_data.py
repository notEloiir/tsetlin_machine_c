from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


def get_mnist_data():
    n_examples = 70000
    n_literals = 784
    
    base_path = Path(__file__).parent / "../data"
    base_path.mkdir(parents=True, exist_ok=True)

    mnist_x_path = base_path / "mnist_x_{}_{}.bin".format(n_examples, n_literals)
    mnist_y_path = base_path / "mnist_y_{}_{}.bin".format(n_examples, n_literals)

    if not (mnist_x_path.exists() and mnist_y_path.exists()):
        print("Fetching MNIST data")
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False)

        x, y = shuffle(X, y, random_state=42)
        x = np.where(x.reshape((x.shape[0], 28 * 28)) > 75, 1, 0)
        x = x.astype(np.uint8)
        y = y.astype(np.uint32)

        assert n_examples == x.shape[0]  # 70000
        assert n_literals == x.shape[1]  # 784
        x.astype(np.uint8).tofile(mnist_x_path)
        y.astype(np.uint32).tofile(mnist_y_path)
    else:
        x = np.fromfile(mnist_x_path, dtype=np.uint8)
        y = np.fromfile(mnist_y_path, dtype=np.uint32)
        x = x.reshape((n_examples, n_literals))

    return x, y


if __name__ == "__main__":
    get_mnist_data()
