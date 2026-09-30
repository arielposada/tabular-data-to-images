"""sklearn-compatible (.fit / .predict) classifiers that wrap the various
tabular-to-image transforms in this repo (DeepInsight, TAC, IGTD, REFINED)
around the shared `SimpleCNN`, so they can be dropped into the same
train/evaluate loop used for the plain sklearn baselines.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pyDeepInsight import ImageTransformer
from pyDeepInsight.utils import Norm2Scaler

from tabular_data_to_images.common.model import SimpleCNN
from tabular_data_to_images.common.train_utils import (
    get_criterion,
    get_device,
    get_optimizer,
    train_epoch,
)
from tabular_data_to_images.IGTD.igtd_transformer import IGTDTransformer
from tabular_data_to_images.tac.feature_vector_transformer import (
    FeatureVectorTransformer,
)
from tabular_data_to_images.tac.load_image import load_and_process_image

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_TAC_BASE_IMAGE = _PACKAGE_DIR / "tac" / "images" / "chess_pattern_36.jpg"
_REFINED_DIR = _PACKAGE_DIR / "refined"
_REFINED_DATASET_DIRS = {"ddos": "ddos", "covid": "covid", "psa": "sismos"}


class _BaseImageClassifier:
    """Shared fit/predict scaffolding: subclasses only need to turn a
    tabular batch into a (N, C, H, W) image array via `_fit_transform`/
    `_transform`; this base class handles the CNN training/inference.
    """

    in_channels = 1
    image_size = 36

    def __init__(self, num_epochs=200, lr=1e-3, batch_size=32, device=None):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device) if device else get_device()

    def _fit_transform(self, X_train, y_train):
        raise NotImplementedError

    def _transform(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        X_img = self._fit_transform(X, y)
        X_t = torch.tensor(X_img, dtype=torch.float32)
        y_t = torch.tensor(y)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        self.model_ = SimpleCNN(
            in_channels=self.in_channels,
            num_classes=n_classes,
            image_size=self.image_size,
        ).to(self.device)
        criterion = get_criterion()
        optimizer = get_optimizer(self.model_, lr=self.lr)

        for epoch in range(self.num_epochs):
            loss, acc = train_epoch(
                self.model_, loader, optimizer, criterion, self.device
            )
            if (epoch + 1) % 50 == 0 or epoch == self.num_epochs - 1:
                print(
                    f"[{type(self).__name__} epoch {epoch + 1}/{self.num_epochs}] "
                    f"loss: {loss:.4f}, acc: {acc:.2f}%"
                )

        return self

    def _forward_batched(self, X):
        X_img = self._transform(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_img, dtype=torch.float32)
        self.model_.eval()
        loader = DataLoader(TensorDataset(X_t), batch_size=self.batch_size, shuffle=False)
        all_logits = []
        with torch.no_grad():
            for (batch,) in loader:
                all_logits.append(self.model_(batch.to(self.device)).cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, X):
        logits = self._forward_batched(X)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        logits = self._forward_batched(X)
        return torch.softmax(logits, dim=1).numpy()


class DeepInsightClassifier(_BaseImageClassifier):
    """Wraps pyDeepInsight's TSNE-based ImageTransformer + SimpleCNN."""

    in_channels = 3

    def __init__(
        self, image_size=36, num_epochs=200, lr=1e-3, batch_size=32, device=None
    ):
        super().__init__(num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=device)
        self.image_size = image_size

    def _fit_transform(self, X_train, y_train):
        self.scaler_ = Norm2Scaler()
        X_norm = self.scaler_.fit_transform(X_train)

        perplexity = min(max(np.sqrt(X_norm.shape[1]), 5), 50)
        reducer = TSNE(
            n_components=2,
            metric="cosine",
            init="random",
            learning_rate="auto",
            perplexity=perplexity,
            n_jobs=-1,
        )
        self.image_transformer_ = ImageTransformer(
            feature_extractor=reducer, pixels=(self.image_size, self.image_size)
        )
        self.image_transformer_.fit(X_norm, y=y_train, plot=False)
        return self._images_to_array(self.image_transformer_.transform(X_norm))

    def _transform(self, X):
        X_norm = self.scaler_.transform(X)
        return self._images_to_array(self.image_transformer_.transform(X_norm))

    @staticmethod
    def _images_to_array(images):
        preprocess = transforms.Compose([transforms.ToTensor()])
        return torch.stack([preprocess(img) for img in images]).numpy()


class TACClassifier(_BaseImageClassifier):
    """Wraps the TAC feature-vector-as-convolution-kernel transform."""

    in_channels = 1
    image_size = 36

    def _fit_transform(self, X_train, y_train):
        self.feature_vector_transformer_ = FeatureVectorTransformer()
        self.feature_vector_transformer_.fit(X_train)
        self.base_image_ = load_and_process_image(_TAC_BASE_IMAGE)  # (1, 36, 36)
        return self._to_images(X_train)

    def _transform(self, X):
        return self._to_images(X)

    def _to_images(self, X):
        kernels = self.feature_vector_transformer_.transform(X)  # (N, j, j)
        kernel_t = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)  # (N,1,j,j)
        padding = (kernel_t.shape[-1] - 1) // 2
        with torch.no_grad():
            # Convolve the single shared base image with each sample's kernel
            # as its own output channel -> (1, N, H, W).
            out = F.conv2d(self.base_image_.unsqueeze(0), kernel_t, padding=padding)
        return out.squeeze(0).unsqueeze(1).numpy()  # (N, 1, H, W)


class IGTDClassifier(_BaseImageClassifier):
    """Wraps the IGTD feature<->pixel mapping transform."""

    in_channels = 1

    def __init__(
        self,
        image_size=36,
        max_step=300,
        val_step=50,
        num_epochs=200,
        lr=1e-3,
        batch_size=32,
        device=None,
        seed=1,
    ):
        super().__init__(num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=device)
        self.image_size = image_size
        self.max_step = max_step
        self.val_step = val_step
        self.seed = seed

    def _fit_transform(self, X_train, y_train):
        self.transformer_ = IGTDTransformer(
            num_row=self.image_size,
            num_col=self.image_size,
            max_step=self.max_step,
            val_step=self.val_step,
            random_state=self.seed,
        )
        self.transformer_.fit(X_train)
        return self._to_images(X_train)

    def _transform(self, X):
        return self._to_images(X)

    def _to_images(self, X):
        images = self.transformer_.transform(X)  # (N, H, W)
        return images[:, None, :, :]


class RefinedClassifier(_BaseImageClassifier):
    """Places features on the pixel grid produced offline by the REFINED
    hill-climbing algorithm (see `refined/Toolbox.py`). Requires a
    precomputed `Mapping_REFINED.pickle` for the given dataset_type -
    REFINED's mapping search itself is not re-run here.
    """

    in_channels = 1

    def __init__(
        self, dataset_type, num_epochs=200, lr=1e-3, batch_size=32, device=None
    ):
        super().__init__(num_epochs=num_epochs, lr=lr, batch_size=batch_size, device=device)
        self.dataset_type = dataset_type

    def _load_mapping(self):
        folder = _REFINED_DATASET_DIRS.get(self.dataset_type)
        if folder is None:
            raise ValueError(
                f"No precomputed REFINED mapping available for dataset "
                f"'{self.dataset_type}'. Known datasets: "
                f"{list(_REFINED_DATASET_DIRS)}"
            )
        mapping_path = _REFINED_DIR / folder / "Mapping_REFINED.pickle"
        with open(mapping_path, "rb") as f:
            gene_names, coords, map_in_int = pickle.load(f)

        self.coords_ = np.asarray(coords)
        self.image_size = int(map_in_int.shape[0])
        self.n_features_ = len(gene_names)

    def _fit_transform(self, X_train, y_train):
        self._load_mapping()
        self.scaler_ = StandardScaler()
        X_norm = self.scaler_.fit_transform(X_train[:, : self.n_features_])
        return self._to_images(X_norm)

    def _transform(self, X):
        X_norm = self.scaler_.transform(X[:, : self.n_features_])
        return self._to_images(X_norm)

    def _to_images(self, X_norm):
        n = X_norm.shape[0]
        images = np.zeros((n, 1, self.image_size, self.image_size), dtype=np.float32)
        images[:, 0, self.coords_[:, 0], self.coords_[:, 1]] = X_norm
        return images
