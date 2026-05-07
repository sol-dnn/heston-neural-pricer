import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from google.colab import drive
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
drive.mount('/content/drive')
print(device)

#data = torch.load("/content/drive/MyDrive/exam_deep_learning/heston_exam_data/heston_raw_dataset.pt")
data = torch.load("/content/drive/MyDrive/exam_deep_learning/heston_exam_data/heston_av_raw_dataset.pt")

X_train_full = data["X_train"].float()
y_train_full = data["y_train"].float().view(-1, 1)
X_test = data["X_test"].float()
y_test = data["y_test"].float().view(-1, 1)

feature_names = data.get("feature_names", ["S0", "Y0", "kappa", "mu", "sigma", "r", "rho", "K", "T"])
print("Train full:", X_train_full.shape, y_train_full.shape)
print("Test      :", X_test.shape, y_test.shape)

val_ratio = 0.10
n_total = len(X_train_full)
n_val = int(val_ratio * n_total)
n_train = n_total - n_val

dataset_full = TensorDataset(X_train_full, y_train_full)
generator = torch.Generator().manual_seed(42)
train_dataset_raw, val_dataset_raw = random_split(dataset_full, [n_train, n_val], generator=generator)

X_train_raw = train_dataset_raw.dataset.tensors[0][train_dataset_raw.indices]
y_train_raw = train_dataset_raw.dataset.tensors[1][train_dataset_raw.indices]

X_val_raw = val_dataset_raw.dataset.tensors[0][val_dataset_raw.indices]
y_val_raw = val_dataset_raw.dataset.tensors[1][val_dataset_raw.indices]

print("Train split:", X_train_raw.shape, y_train_raw.shape)
print("Val split  :", X_val_raw.shape, y_val_raw.shape)

# Standardize X using TRAIN statistics
x_mean = X_train_raw.mean(dim=0, keepdim=True)
x_std = X_train_raw.std(dim=0, keepdim=True).clamp_min(1e-8)
X_train = (X_train_raw - x_mean) / x_std
X_val = (X_val_raw - x_mean) / x_std
X_test_std = (X_test - x_mean) / x_std
y_train = y_train_raw
y_val = y_val_raw
y_test_std_target = y_test

# DataLoaders
batch_size = 1024
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test_std, y_test_std_target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def full_metrics(y_pred, y_true, min_price=0.5):
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    mask_atm = y_true > min_price
    mask_otm = ~mask_atm

    abs_err = torch.abs(y_pred - y_true)
    err_np  = (y_pred - y_true).cpu().numpy()
    abs_np  = abs_err.cpu().numpy()

    def rel(pred, true):
        return (torch.abs(pred - true) / (true + 1e-3)).mean().item()

    def rel2(pred, true):
      return (torch.abs(pred - true) / (true)).mean().item()

    return {
        # Global
        "mae":           abs_err.mean().item(),
        "rmse":          torch.sqrt(((y_pred - y_true)**2).mean()).item(),
        "rel_err_total": rel(y_pred, y_true),
        "rel_err":   rel2(y_pred[mask_atm], y_true[mask_atm]) if mask_atm.sum() > 0 else float('nan'),

        # Distribution
        "error_mean":    float(err_np.mean()),
        "error_std":     float(err_np.std()),
        "abs_p50":       float(np.percentile(abs_np, 50)),
        "abs_p90":       float(np.percentile(abs_np, 90)),
        "abs_p95":       float(np.percentile(abs_np, 95)),
        "abs_p99":       float(np.percentile(abs_np, 99)),
    }

'''
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.5, eps=1.0):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        weights = torch.log1p(target / self.eps + 1.0)
        huber = F.huber_loss(pred, target, delta=self.delta, reduction='none')
        return (weights * huber).mean()
'''

class HestonMLP(nn.Module):
    def __init__(self, input_dim=9, dropout=0.05):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(dropout)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop4(x)

        x = self.out(x)
        return x

@torch.no_grad()
def evaluate_model(model, loader, device=device):
    model.eval()
    preds_all, targets_all = [], []
    for xb, yb in loader:
        preds_all.append(model(xb.to(device)))
        targets_all.append(yb.to(device))
    preds_all  = torch.cat(preds_all).cpu()
    targets_all = torch.cat(targets_all).cpu()
    return full_metrics(preds_all, targets_all), preds_all, targets_all


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=20,
    lr=1e-3,
    weight_decay=1e-5,
    patience=12,
    grad_clip=None,
    device=device
):
    model = model.to(device)

    criterion = nn.HuberLoss(delta=0.5)
    #criterion = WeightedHuberLoss(delta=0.5, eps=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5
    )

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_rel_err": [],
        "lr": [],
    }

    best_model_state = None
    best_val_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()

        running_loss = 0.0
        running_mae_sum = 0.0
        running_n = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)

        for xb, yb in progress_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            batch_n = xb.size(0)
            running_loss += loss.item() * batch_n
            running_mae_sum += torch.sum(torch.abs(pred.detach() - yb)).item()
            running_n += batch_n

            progress_bar.set_postfix({
                "train_loss": f"{running_loss / running_n:.5f}",
                "train_mae": f"{running_mae_sum / running_n:.5f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        train_loss = running_loss / running_n
        train_mae = running_mae_sum / running_n

        val_metrics, _, _ = evaluate_model(model, val_loader, device=device)
        scheduler.step(val_metrics["mae"])

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_mae)
        history["val_loss"].append(val_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_rel_err"].append(val_metrics["rel_err_total"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | train_mae={train_mae:.6f} | "
            f"val_mae={val_metrics['mae']:.6f} | val_rmse={val_metrics['rmse']:.6f} | "
            f"val_rel_total={val_metrics['rel_err_total']:.6f} | val_rel_atm={val_metrics['rel_err']:.6f} | "
            f"lr={current_lr:.2e}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

model = HestonMLP(input_dim=9, dropout=0.05).to(device)
print(model)

model, history = train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    grad_clip=None,
    device=device
)

train_metrics, train_preds, train_targets = evaluate_model(model, train_loader, device=device)
val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, device=device)
test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, device=device)

print("\nFinal metrics")
print("TRAIN:", train_metrics)
print("VAL  :", val_metrics)
print("TEST :", test_metrics)

# Save model + preprocessing stats
save_dict = {
    "model_state_dict": model.state_dict(),
    "x_mean": x_mean,
    "x_std": x_std,
    "feature_names": feature_names,
    "history": history,
    "model_config": {
        "input_dim": 9,
        "hidden_dims": (256, 256, 128, 64),
        "dropout": 0.05
    }
}
torch.save(save_dict, "/content/drive/MyDrive/exam_deep_learning/heston_exam_data/heston_mlp_model.pt")
print("Model saved.")

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training and validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_pred, y_true, title="Error distribution"):
    err = (y_pred - y_true).detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.hist(err, bins=80)
    plt.xlabel("Prediction error")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pred_vs_true(y_pred, y_true, title="Predicted vs true prices"):
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    y_true_np = y_true.detach().cpu().numpy().reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_np, y_pred_np, s=5, alpha=0.4)
    mn = min(y_true_np.min(), y_pred_np.min())
    mx = max(y_true_np.max(), y_pred_np.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True price")
    plt.ylabel("Predicted price")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_histograms(y_pred, y_true, X_raw, min_price=0.25):
    y_pred  = y_pred.detach().cpu().numpy().reshape(-1)
    y_true  = y_true.detach().cpu().numpy().reshape(-1)

    mask = y_true > min_price
    S0 = X_raw[:, 0].cpu().numpy()
    K  = X_raw[:, 7].cpu().numpy()
    mask_otm = S0 < K

    abs_err = np.abs(y_pred - y_true)
    rel_err = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]  # pas d'epsilon

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Error distribution — {mask.sum()} contracts with price > {min_price}")
    axes[0].hist(abs_err, bins=100, edgecolor='none', color='steelblue', label='All')
    axes[0].axvline(np.median(abs_err), color='k', linestyle='--', label=f'Median={np.median(abs_err):.4f}')
    axes[0].set_xlabel("Absolute error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Absolute error (all contracts)")
    axes[0].legend()

    axes[1].hist(rel_err, bins=100, edgecolor='none', color='steelblue')
    axes[1].axvline(np.median(rel_err), color='r', linestyle='--', label=f'Median={np.median(rel_err):.4f}')
    axes[1].axvline(np.percentile(rel_err, 95), color='orange', linestyle='--', label=f'P95={np.percentile(rel_err, 95):.4f}')
    axes[1].set_xlabel("Relative error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_ylim(0, 1000)
    axes[1].set_title(f"Relative error (price > {min_price})")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

plot_error_histograms(test_preds, test_targets, X_test, min_price=0.5)
plot_training_history(history)
plot_error_distribution(test_preds, test_targets, title="Test error distribution")
plot_pred_vs_true(test_preds, test_targets, title="Test: predicted vs true prices")

MIN_PRICE_REL=0.5

@torch.no_grad()
def get_predictions_dataframe(model, X_raw, y_true, x_mean, x_std, feature_names, device, min_price_rel=MIN_PRICE_REL):
    """
    X_raw : raw (non-standardized) inputs, shape (N, 9)
    y_true: true prices, shape (N,) or (N,1)
    """
    model.eval()

    X_std = (X_raw - x_mean) / x_std
    X_std = X_std.to(device)
    y_pred = model(X_std).detach().cpu().reshape(-1)
    y_true = y_true.detach().cpu().reshape(-1)

    X_np = X_raw.detach().cpu().numpy()
    df = pd.DataFrame(X_np, columns=feature_names)

    df["y_true"] = y_true.numpy()
    df["y_pred"] = y_pred.numpy()
    df["abs_error"] = np.abs(df["y_pred"] - df["y_true"])
    df["signed_error"] = df["y_pred"] - df["y_true"]

    # raw relative error stored but should only be analyzed on filtered subset
    df["rel_error"] = df["abs_error"] / (np.abs(df["y_true"]) + 1e-6)
    df["moneyness"] = df["S0"] / df["K"]
    df["rel_mask"] = df["y_true"] > min_price_rel

    return df


def filter_relative_error_df(df, min_price_rel=MIN_PRICE_REL):
    return df[df["y_true"] > min_price_rel].copy()


# MONEyness BUCKETS
def add_moneyness_bucket(df, atm_band=0.02):
    """
    ATM if 1-atm_band <= S0/K <= 1+atm_band
    """
    m = df["moneyness"]

    conditions = [
        m < 1.0 - atm_band,
        (m >= 1.0 - atm_band) & (m <= 1.0 + atm_band),
        m > 1.0 + atm_band
    ]
    labels = ["OTM", "ATM", "ITM"]

    df = df.copy()
    df["contract_type"] = np.select(conditions, labels, default="ATM")
    return df



# CONTRACT-TYPE PLOTS
def plot_error_by_contract_type(df, error_col="abs_error", atm_band=0.02):
    df_plot = add_moneyness_bucket(df, atm_band=atm_band)

    order = ["OTM", "ATM", "ITM"]
    grouped = df_plot.groupby("contract_type")[error_col]

    means = [grouped.mean().get(k, np.nan) for k in order]
    medians = [grouped.median().get(k, np.nan) for k in order]
    p90 = [grouped.quantile(0.9).get(k, np.nan) for k in order]

    x = np.arange(len(order))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, means, width=width, label="Mean")
    plt.bar(x, medians, width=width, label="Median")
    plt.bar(x + width, p90, width=width, label="90th pct")
    plt.xticks(x, order)
    plt.ylabel(error_col.replace("_", " ").title())
    plt.title(f"{error_col.replace('_', ' ').title()} by contract type")
    plt.legend()
    plt.tight_layout()
    plt.show()


def boxplot_error_by_contract_type(df, error_col="abs_error", atm_band=0.02):
    df_plot = add_moneyness_bucket(df, atm_band=atm_band)

    order = ["OTM", "ATM", "ITM"]
    data = [df_plot.loc[df_plot["contract_type"] == k, error_col].values for k in order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=order, showfliers=False)
    plt.ylabel(error_col.replace("_", " ").title())
    plt.title(f"Distribution of {error_col.replace('_', ' ')} by contract type")
    plt.tight_layout()
    plt.show()


# DECILE CURVE WITH CONFIDENCE INTERVAL
# abs_error: use full df ; rel_error: use filtered df only
def compute_decile_curve(df, feature_col, error_col, n_bins=10, ci_level=0.95):
    tmp = df[[feature_col, error_col]].copy()
    tmp["bin"] = pd.qcut(tmp[feature_col], q=n_bins, duplicates="drop")

    z = 1.96 if ci_level == 0.95 else 1.96

    def summarize_bin(g):
        x = g[feature_col].values
        e = g[error_col].values
        n = len(e)

        error_mean = np.mean(e)
        error_std = np.std(e, ddof=1) if n > 1 else 0.0
        error_se = error_std / np.sqrt(n) if n > 1 else 0.0

        mean_ci_low = error_mean - z * error_se
        mean_ci_high = error_mean + z * error_se

        return pd.Series({
            "feature_mean": np.mean(x),
            "feature_median": np.median(x),
            "error_mean": error_mean,
            "mean_ci_low": mean_ci_low,
            "mean_ci_high": mean_ci_high,
            "count": n
        })

    grouped = (
        tmp.groupby("bin", observed=False)
           .apply(summarize_bin)
           .reset_index(drop=True)
    )

    return grouped



def plot_feature_abs_rel_side_by_side(df, feature_col, n_bins=10, ci_level=0.95, min_price_rel=MIN_PRICE_REL):
    df_abs = df.copy()
    df_rel = df[df["y_true"] > min_price_rel].copy()

    grouped_abs = compute_decile_curve(df_abs, feature_col, "abs_error", n_bins=n_bins, ci_level=ci_level)
    grouped_rel = compute_decile_curve(df_rel, feature_col, "rel_error", n_bins=n_bins, ci_level=ci_level)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x_abs = grouped_abs["feature_mean"].values
    axes[0].plot(x_abs, grouped_abs["error_mean"].values, marker="o", linewidth=2, label="Mean")
    axes[0].fill_between(
        x_abs,
        grouped_abs["mean_ci_low"].values,
        grouped_abs["mean_ci_high"].values,
        alpha=0.2,
        label=f"Mean ± {int(ci_level*100)}% CI"
    )
    axes[0].set_xlabel(f"Mean {feature_col} within decile")
    axes[0].set_ylabel("Absolute error")
    axes[0].set_title(f"Absolute error vs {feature_col}")
    axes[0].legend()

    x_rel = grouped_rel["feature_mean"].values
    axes[1].plot(x_rel, grouped_rel["error_mean"].values, marker="o", linewidth=2, label="Mean")
    axes[1].fill_between(
        x_rel,
        grouped_rel["mean_ci_low"].values,
        grouped_rel["mean_ci_high"].values,
        alpha=0.2,
        label=f"Mean ± {int(ci_level*100)}% CI"
    )
    axes[1].set_xlabel(f"Mean {feature_col} within decile")
    axes[1].set_ylabel(f"Relative error (y_true > {min_price_rel})")
    axes[1].set_title(f"Relative error vs {feature_col}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_all_features_abs_rel_side_by_side(df, n_bins=10, ci_level=0.95, min_price_rel=MIN_PRICE_REL):
    feature_cols = ["moneyness", "S0", "Y0", "kappa", "mu", "sigma", "r", "rho", "K", "T"]

    for feature_col in feature_cols:
        plot_feature_abs_rel_side_by_side(
            df=df,
            feature_col=feature_col,
            n_bins=n_bins,
            ci_level=ci_level,
            min_price_rel=min_price_rel
        )

df_test = get_predictions_dataframe(
    model=model,
    X_raw=X_test,
    y_true=test_targets,
    x_mean=x_mean,
    x_std=x_std,
    feature_names=feature_names,
    device=device,
    min_price_rel=MIN_PRICE_REL
)
df_test_rel = filter_relative_error_df(df_test, min_price_rel=MIN_PRICE_REL)

df_test_rel

plot_error_by_contract_type(df_test, error_col="abs_error")
boxplot_error_by_contract_type(df_test, error_col="abs_error")

plot_error_by_contract_type(df_test_rel, error_col="rel_error")
boxplot_error_by_contract_type(df_test_rel, error_col="rel_error")

plot_all_features_abs_rel_side_by_side(df_test, n_bins=10, ci_level=0.95, min_price_rel=MIN_PRICE_REL)

"""# Comparison of accuracy with larger model (more neuron and more layers)"""

class HestonMLPLarge(nn.Module):
    def __init__(self, input_dim=9, dropout=0.05):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.drop6 = nn.Dropout(dropout)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.drop5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.drop6(x)

        x = self.out(x)
        return x

model_large = HestonMLPLarge(input_dim=9, dropout=0.05).to(device)
print(model_large)

model, history = train_model(
    model_large,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    grad_clip=None,
    device=device
)

train_metrics, train_preds, train_targets = evaluate_model(model, train_loader, device=device)
val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, device=device)
test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, device=device)

print("\nFinal metrics")
print("TRAIN:", train_metrics)
print("VAL  :", val_metrics)
print("TEST :", test_metrics)

class HestonMLPSmall(nn.Module):
    def __init__(self, input_dim=9, dropout=0.05):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(dropout)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)

        x = self.out(x)
        return x

model_small = HestonMLPSmall(input_dim=9, dropout=0.05).to(device)
print(model_small)

model, history = train_model(
    model_small,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    grad_clip=None,
    device=device
)

train_metrics, train_preds, train_targets = evaluate_model(model, train_loader, device=device)
val_metrics, val_preds, val_targets = evaluate_model(model, val_loader, device=device)
test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, device=device)

print("\nFinal metrics")
print("TRAIN:", train_metrics)
print("VAL  :", val_metrics)
print("TEST :", test_metrics)

class HestonMLPVeryLarge(nn.Module):
    def __init__(self, input_dim=9, dropout=0.05):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.drop5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.drop6 = nn.Dropout(dropout)

        self.fc7 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.drop7 = nn.Dropout(dropout)

        self.fc8 = nn.Linear(64, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.drop8 = nn.Dropout(dropout)

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.drop4(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.drop5(x)

        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.drop7(x)

        x = self.fc8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.drop8(x)

        x = self.out(x)
        return x

model_verylarge = HestonMLPVeryLarge(input_dim=9, dropout=0.05).to(device)
print(model_verylarge)


model_verylarge, history = train_model(
    model_verylarge,
    train_loader,
    val_loader,
    n_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    patience=15,
    grad_clip=None,
    device=device
)


train_metrics, train_preds, train_targets = evaluate_model(model_verylarge, train_loader, device=device)
val_metrics, val_preds, val_targets = evaluate_model(model_verylarge, val_loader, device=device)
test_metrics, test_preds, test_targets = evaluate_model(model_verylarge, test_loader, device=device)

print("\nFinal metrics")
print("TRAIN:", train_metrics)
print("VAL  :", val_metrics)
print("TEST :", test_metrics)