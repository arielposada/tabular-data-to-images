"""Fit/transform wrapper around the IGTD algorithm (see igtd_functions.py),
mirroring the fit/transform shape of pyDeepInsight's ImageTransformer used
elsewhere in this project.
"""

import pickle

import numpy as np

from tabular_data_to_images.IGTD.igtd_functions import (
    generate_feature_distance_ranking,
    generate_image_data,
    generate_matrix_distance_ranking,
    min_max_transform,
    IGTD_absolute_error,
)


class IGTDTransformer:
    def __init__(
        self,
        num_row,
        num_col,
        fea_dist_method="Pearson",
        image_dist_method="Euclidean",
        max_step=300,
        val_step=50,
        random_state=1,
        save_folder=None,
    ):
        self.num_row = num_row
        self.num_col = num_col
        self.fea_dist_method = fea_dist_method
        self.image_dist_method = image_dist_method
        self.max_step = max_step
        self.val_step = val_step
        self.random_state = random_state
        self.save_folder = save_folder

        self.index_ = None
        self.coord_ = None
        self.err_record_ = None
        self.min_v_ = None
        self.max_v_ = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        self.min_v_ = np.nanmin(X_train, axis=0)
        self.max_v_ = np.nanmax(X_train, axis=0)

        norm_d = min_max_transform(X_train)

        ranking_feature, _ = generate_feature_distance_ranking(
            norm_d, method=self.fea_dist_method
        )
        coord, ranking_image = generate_matrix_distance_ranking(
            num_r=self.num_row,
            num_c=self.num_col,
            method=self.image_dist_method,
            num=norm_d.shape[1],
        )

        index_record, err_record, _ = IGTD_absolute_error(
            source=ranking_feature,
            target=ranking_image,
            max_step=self.max_step,
            val_step=self.val_step,
            random_state=self.random_state,
            save_folder=self.save_folder,
        )

        best_step = int(np.argmin(err_record))
        self.index_ = index_record[best_step, :]
        self.coord_ = coord
        self.err_record_ = err_record
        return self

    def _normalize(self, X):
        X = np.asarray(X, dtype=np.float64)
        denom = self.max_v_ - self.min_v_
        denom[denom == 0] = 1.0
        norm = (X - self.min_v_) / denom
        return np.clip(norm, 0.0, 1.0)

    def transform(self, X):
        if self.index_ is None:
            raise RuntimeError(
                "IGTDTransformer.fit() must be called before transform()."
            )
        norm_d = self._normalize(X)
        images = generate_image_data(
            norm_d,
            index=self.index_,
            num_row=self.num_row,
            num_column=self.num_col,
            coord=self.coord_,
        )
        # (num_row, num_column, n_samples) -> (n_samples, num_row, num_column)
        return np.transpose(images, (2, 0, 1))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index_,
                    "coord": self.coord_,
                    "num_row": self.num_row,
                    "num_col": self.num_col,
                    "min_v": self.min_v_,
                    "max_v": self.max_v_,
                    "err_record": self.err_record_,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(num_row=state["num_row"], num_col=state["num_col"])
        obj.index_ = state["index"]
        obj.coord_ = state["coord"]
        obj.min_v_ = state["min_v"]
        obj.max_v_ = state["max_v"]
        obj.err_record_ = state["err_record"]
        return obj
