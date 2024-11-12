import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)  # Adjust input size according to your features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Server(fl.server.Server):
    def __init__(self):
        super().__init__()
        self.model = TitanicModel()

    def get_parameters(self, config=None) -> List[np.ndarray]:
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).to(param.data.device)

    def aggregate_fit(self, results: List[Tuple[str, List[np.ndarray]]]) -> List[np.ndarray]:
        num_samples = sum(num_samples for _, num_samples in results)
        new_parameters = [np.zeros_like(param) for param in self.get_parameters()]

        for idx, (parameters, num_samples) in enumerate(results):
            for i, param in enumerate(parameters):
                new_parameters[i] += param * num_samples / num_samples

        return new_parameters

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        return float(val_loss), len(val_data), {"accuracy": accuracy}


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate fit metrics from clients."""
    aggregated_metrics = {}
    total_samples = sum(num_samples for num_samples, _ in metrics)

    for num_samples, metric_dict in metrics:
        for key, value in metric_dict.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * num_samples / total_samples

    return aggregated_metrics

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Aggregate evaluation metrics from clients."""
    aggregated_metrics = {}
    total_samples = sum(num_samples for num_samples, _ in metrics)

    for num_samples, metric_dict in metrics:
        for key, value in metric_dict.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * num_samples / total_samples

    return aggregated_metrics


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,  # Minimum number of clients to train on each round
    min_available_clients=2,  # Minimum number of clients to be connected
    on_fit_config_fn=lambda rnd: {"epoch": rnd},
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
