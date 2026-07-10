import torch  # noqa: F401  (import first: avoids a numpy/torch DLL init clash on Windows)
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from sklearn.metrics import f1_score
import optuna

from tabular_data_to_images.upsampling_approach.model import CombinedModel
from tabular_data_to_images.common.dataset import (
    load_dataset,
    load_psa_data,
    split_data,
)
from tabular_data_to_images.common.train_utils import (
    get_device,
    get_optimizer,
    get_criterion,
    train_epoch,
    evaluate_model,
)


def optimize_model(dataset_type="psa"):
    # Load and prepare data
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=True
    )

    device = get_device()
    input_size = X_train.shape[1]

    # Define the objective function to minimize
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = CombinedModel(input_size, num_classes).to(device)
        optimizer = get_optimizer(model, lr)
        criterion = get_criterion()

        best_f1_score = 0.0
        for epoch in tqdm(range(50)):
            train_epoch(model, trainloader, optimizer, criterion, device)
            y_test_true, y_test_pred = evaluate_model(model, testloader, device)
            val_f1 = f1_score(y_test_true, y_test_pred, average="weighted")
            if val_f1 > best_f1_score:
                best_f1_score = val_f1

            # Handle trial pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_f1_score

    # Create an Optuna study and run optimization
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=50)

    print(f"Best parameters: {study.best_params}")


if __name__ == "__main__":
    # optimize_model("psa")
    optimize_model("ddos")
