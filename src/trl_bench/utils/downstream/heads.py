"""
Unified downstream model heads.

Provides both PyTorch nn.Module heads (MLPHead, DualProjectionHead) and
sklearn-compatible heads (get_sklearn_head) under a single namespace.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# ============================================================================
# Activation mapping
# ============================================================================

ACTIVATIONS = {
    "relu": nn.ReLU,
}


# ============================================================================
# Marker base
# ============================================================================

class DownstreamHead:
    """Common interface marker. All heads implement either:
    - nn.Module interface: forward(x) -> logits  (PyTorch heads)
    - sklearn interface: fit(X, y) / predict(X)  (sklearn/xgboost heads)
    """
    pass


# ============================================================================
# Adapter
# ============================================================================

class AdapterLayer(nn.Module):
    """Bottleneck adapter: down -> activation -> up + residual -> LayerNorm."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 64, activation: str = "relu"):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        self.act = ACTIVATIONS.get(activation, nn.ReLU)()
        self.norm = nn.LayerNorm(input_dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.up(self.act(self.down(x))))


# ============================================================================
# MLPHead — unified PyTorch head
# ============================================================================

class MLPHead(nn.Module, DownstreamHead):
    """
    Unified MLP head for all downstream tasks.

    Architecture: input -> Linear(input_dim, 256) -> ReLU -> Dropout -> Linear(256, output_dim).
    Supports dropout-first mode, normalization, residual connections, and adapters.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.1,
        dropout_first: bool = False,
        norm_type: str = "none",
        normalize_output: bool = False,
        use_residual: bool = False,
        use_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.dropout = dropout
        self.dropout_first = dropout_first
        self.norm_type = norm_type
        self.normalize_output = normalize_output
        self.use_residual = use_residual
        self.use_adapter = use_adapter
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")
        act_fn = ACTIVATIONS[activation]

        layers = []

        # Optional leading dropout (TabFact / CT style)
        if dropout_first:
            layers.append(nn.Dropout(dropout))

        if self.num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First hidden layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            if norm_type == "layer_norm":
                layers.append(nn.LayerNorm(hidden_dim))
            elif norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))

            # Middle hidden layers
            for _ in range(self.num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm_type == "layer_norm":
                    layers.append(nn.LayerNorm(hidden_dim))
                elif norm_type == "batch_norm":
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Residual projection if dimensions don't match
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

        # Optional adapter
        if use_adapter:
            self.adapter = AdapterLayer(output_dim, adapter_bottleneck_dim, activation)
        else:
            self.adapter = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)

        if self.use_residual:
            residual = self.residual_proj(x) if self.residual_proj is not None else x
            out = out + residual

        if self.adapter is not None:
            out = self.adapter(out)

        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def to_config(self) -> Dict[str, Any]:
        return {
            'type': 'mlp',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'activation': self.activation_name,
            'dropout': self.dropout,
            'dropout_first': self.dropout_first,
            'norm_type': self.norm_type,
            'normalize_output': self.normalize_output,
            'use_residual': self.use_residual,
            'use_adapter': self.use_adapter,
            'adapter_bottleneck_dim': self.adapter_bottleneck_dim,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> 'MLPHead':
        params = {k: v for k, v in cfg.items() if k not in ('type', 'hidden_dims')}
        return cls(**params)


# ============================================================================
# DualProjectionHead
# ============================================================================

class DualProjectionHead(nn.Module, DownstreamHead):
    """Dual projection heads for bi-encoder retrieval (table + query)."""

    def __init__(
        self,
        table_input_dim: int = 768,
        query_input_dim: int = 768,
        output_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        share_projection: bool = False,
        num_layers: int = 2,
        activation: str = "relu",
        norm_type: str = "none",
        normalize_output: bool = False,
        use_residual: bool = False,
        use_adapter: bool = False,
        adapter_bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.share_projection = share_projection

        self._config = {
            'table_input_dim': table_input_dim,
            'query_input_dim': query_input_dim,
            'output_dim': output_dim,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'share_projection': share_projection,
            'num_layers': num_layers,
            'activation': activation,
            'norm_type': norm_type,
            'normalize_output': normalize_output,
            'use_residual': use_residual,
            'use_adapter': use_adapter,
            'adapter_bottleneck_dim': adapter_bottleneck_dim,
        }

        proj_kwargs = dict(
            output_dim=output_dim, hidden_dim=hidden_dim, dropout=dropout,
            num_layers=num_layers, activation=activation, norm_type=norm_type,
            normalize_output=normalize_output, use_residual=use_residual,
            use_adapter=use_adapter, adapter_bottleneck_dim=adapter_bottleneck_dim,
        )

        self.table_proj = MLPHead(input_dim=table_input_dim, **proj_kwargs)

        if share_projection:
            self.query_proj = self.table_proj
        else:
            self.query_proj = MLPHead(input_dim=query_input_dim, **proj_kwargs)

    def forward_table(self, table_emb: torch.Tensor) -> torch.Tensor:
        return self.table_proj(table_emb)

    def forward_query(self, query_emb: torch.Tensor) -> torch.Tensor:
        return self.query_proj(query_emb)

    def forward(
        self,
        table_emb: Optional[torch.Tensor] = None,
        query_emb: Optional[torch.Tensor] = None,
    ) -> tuple:
        table_proj = self.forward_table(table_emb) if table_emb is not None else None
        query_proj = self.forward_query(query_emb) if query_emb is not None else None
        return table_proj, query_proj

    def save(self, path: str):
        torch.save({
            'table_proj': self.table_proj.state_dict(),
            'query_proj': self.query_proj.state_dict() if not self.share_projection else None,
            'config': self._config,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'DualProjectionHead':
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']

        defaults = {
            'num_layers': 2, 'activation': 'relu', 'norm_type': 'none',
            'normalize_output': False, 'use_residual': False, 'dropout': 0.1,
            'use_adapter': False, 'adapter_bottleneck_dim': 64,
        }
        for key, default_val in defaults.items():
            if key not in config:
                config[key] = default_val

        if 'hidden_dim' not in config:
            # Backward compat: infer from checkpoint weights
            # The MLPHead stores layers in self.mlp; first linear bias shape gives hidden_dim
            table_sd = checkpoint['table_proj']
            for k, v in table_sd.items():
                if k.startswith('mlp.') and k.endswith('.bias'):
                    config['hidden_dim'] = v.shape[0]
                    break

        config.pop('hidden_dims', None)
        model = cls(**config)
        model.table_proj.load_state_dict(checkpoint['table_proj'])
        if not config['share_projection'] and checkpoint['query_proj'] is not None:
            model.query_proj.load_state_dict(checkpoint['query_proj'])

        return model.to(device)

    def to_config(self) -> Dict[str, Any]:
        cfg = dict(self._config)
        cfg['type'] = 'dual_projection'
        return cfg

    @classmethod
    def from_config(cls, cfg: dict) -> 'DualProjectionHead':
        params = {k: v for k, v in cfg.items() if k not in ('type', 'hidden_dims')}
        return cls(**params)


# ============================================================================
# ProjectedInteractionHead
# ============================================================================

class ProjectedInteractionHead(nn.Module, DownstreamHead):
    """
    Projected interaction head for cross-modal classification.

    Designed for tasks where two independently-encoded modalities (e.g., table
    embeddings and statement embeddings) must be compared for relationship
    reasoning (e.g., entailment vs refutation in fact verification).

    Architecture:
        1. Split concatenated input into table and statement embeddings
        2. Project each through independent linear projections + LayerNorm
        3. Build interaction features: [t; s; t*s; |t-s|]
        4. Classify with a small MLP

    The forward(x) signature accepts a single concatenated tensor, so this head
    is a drop-in replacement for MLPHead in existing pipelines that use
    combine_method='concat'.
    """

    def __init__(
        self,
        table_input_dim: int = 768,
        stmt_input_dim: int = 768,
        projection_dim: int = 256,
        classifier_hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        interaction_type: str = 'full',
        normalize_projection: bool = True,
    ):
        super().__init__()
        self.table_input_dim = table_input_dim
        self.stmt_input_dim = stmt_input_dim
        self.projection_dim = projection_dim
        self.interaction_type = interaction_type
        self.normalize_projection = normalize_projection

        self._config = {
            'table_input_dim': table_input_dim,
            'stmt_input_dim': stmt_input_dim,
            'projection_dim': projection_dim,
            'classifier_hidden_dim': classifier_hidden_dim,
            'num_classes': num_classes,
            'dropout': dropout,
            'interaction_type': interaction_type,
            'normalize_projection': normalize_projection,
        }

        # Projection towers (linear only — keep shallow per design)
        self.table_proj = nn.Linear(table_input_dim, projection_dim)
        self.stmt_proj = nn.Linear(stmt_input_dim, projection_dim)

        # Optional LayerNorm after projection
        if normalize_projection:
            self.table_norm = nn.LayerNorm(projection_dim)
            self.stmt_norm = nn.LayerNorm(projection_dim)
        else:
            self.table_norm = None
            self.stmt_norm = None

        # Interaction feature dimension
        if interaction_type == 'full':
            interaction_dim = 4 * projection_dim  # [t; s; t*s; |t-s|]
        elif interaction_type == 'relation_only':
            interaction_dim = 2 * projection_dim  # [t*s; |t-s|]
        else:
            raise ValueError(
                f"Unknown interaction_type: {interaction_type}. "
                f"Use 'full' or 'relation_only'."
            )

        # Classifier MLP (reuses existing MLPHead)
        self.classifier = MLPHead(
            input_dim=interaction_dim,
            output_dim=num_classes,
            hidden_dim=classifier_hidden_dim,
            num_layers=2,
            activation='relu',
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split concatenated input into table and statement parts
        table_emb = x[:, :self.table_input_dim]
        stmt_emb = x[:, self.table_input_dim:]

        # Project into shared space
        t = self.table_proj(table_emb)
        s = self.stmt_proj(stmt_emb)

        # Normalize
        if self.normalize_projection:
            t = self.table_norm(t)
            s = self.stmt_norm(s)

        # Build interaction features
        product = t * s               # element-wise product (alignment signal)
        diff = torch.abs(t - s)       # absolute difference (divergence signal)

        if self.interaction_type == 'full':
            features = torch.cat([t, s, product, diff], dim=-1)
        else:
            features = torch.cat([product, diff], dim=-1)

        return self.classifier(features)

    def to_config(self) -> Dict[str, Any]:
        cfg = dict(self._config)
        cfg['type'] = 'projected_interaction'
        return cfg

    @classmethod
    def from_config(cls, cfg: dict) -> 'ProjectedInteractionHead':
        params = {k: v for k, v in cfg.items() if k not in ('type',)}
        return cls(**params)


# ============================================================================
# build_head factory
# ============================================================================

def build_head(config: dict) -> nn.Module:
    """Build a PyTorch head from a config dict. Dispatches on config['type']."""
    head_type = config.get('type', 'mlp')
    if head_type == 'mlp':
        return MLPHead.from_config(config)
    elif head_type == 'dual_projection':
        return DualProjectionHead.from_config(config)
    elif head_type == 'projected_interaction':
        return ProjectedInteractionHead.from_config(config)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


# ============================================================================
# Sklearn-compatible heads (MLPNet, TorchMLP*, factories)
# ============================================================================

class MLPNet(nn.Module):
    """Deprecated: use MLPHead instead.

    Kept for backward compatibility with pickled model files that
    reference this class (self.network attribute).
    """

    def __init__(self, input_dim, hidden_dims=(256, 128), output_dim=1, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier(DownstreamHead):
    """sklearn-compatible PyTorch MLP Classifier with GPU support."""

    def __init__(self, hidden_dim=256, max_iter=500,
                 random_state=42, early_stopping=True, validation_fraction=0.1,
                 use_gpu=True, batch_size=512, learning_rate=0.001):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = None
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.device = torch.device('cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu')

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([label_to_idx[label] for label in y])

        if self.early_stopping:
            n_samples = len(X)
            n_val = int(n_samples * self.validation_fraction)
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]
            X_train, y_train = X[train_idx], y_mapped[train_idx]
            X_val, y_val = X[val_idx], y_mapped[val_idx]
        else:
            X_train, y_train = X, y_mapped
            X_val, y_val = None, None

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)

        input_dim = X.shape[1]
        self.model = MLPHead(
            input_dim=input_dim,
            output_dim=self.n_classes_,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        pbar = tqdm(range(self.max_iter), desc='Training MLP Classifier')

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if self.early_stopping and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'patience': f'{patience_counter}/{patience}'
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'val_loss': f'{val_loss:.4f}',
                            'status': 'early stopped'
                        })
                        break
            else:
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        pbar.close()
        return self

    def predict(self, X):
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        return self.classes_[predictions]

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()
        return proba


class TorchMLPRegressor(DownstreamHead):
    """sklearn-compatible PyTorch MLP Regressor with GPU support."""

    def __init__(self, hidden_dim=256, max_iter=500,
                 random_state=42, early_stopping=True, validation_fraction=0.1,
                 use_gpu=True, batch_size=512, learning_rate=0.001):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.device = None

    def fit(self, X, y):
        self.device = torch.device('cuda' if (self.use_gpu and torch.cuda.is_available()) else 'cpu')

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if self.early_stopping:
            n_samples = len(X)
            n_val = int(n_samples * self.validation_fraction)
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
            train_idx = indices[n_val:]
            val_idx = indices[:n_val]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        input_dim = X.shape[1]
        self.model = MLPHead(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        pbar = tqdm(range(self.max_iter), desc='Training MLP Regressor')

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if self.early_stopping and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'patience': f'{patience_counter}/{patience}'
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'val_loss': f'{val_loss:.4f}',
                            'status': 'early stopped'
                        })
                        break
            else:
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        pbar.close()
        return self

    def predict(self, X):
        self.model.eval()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        return predictions


# ============================================================================
# Sklearn head factory
# ============================================================================

def get_sklearn_head(name: str, task: str = "classification", use_gpu: bool = True, **kwargs):
    """
    Factory returning a sklearn-compatible model by name.

    Args:
        name: 'mlp', 'xgboost', 'random_forest', 'logistic', 'svm',
              'linear', 'ridge', 'lasso'
        task: 'classification' or 'regression'
        use_gpu: Whether to use GPU for models that support it
        **kwargs: Additional keyword arguments passed to the model constructor

    Returns:
        A sklearn-compatible model instance with fit/predict interface.
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC, SVR
    import xgboost as xgb

    gpu_available = torch.cuda.is_available()
    device = 'cuda' if (use_gpu and gpu_available) else 'cpu'

    if task == "classification":
        models = {
            'mlp': lambda: TorchMLPClassifier(
                hidden_dim=kwargs.get('hidden_dim', 256),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42),
                early_stopping=kwargs.get('early_stopping', True),
                validation_fraction=kwargs.get('validation_fraction', 0.1),
                use_gpu=use_gpu,
            ),
            'logistic': lambda: LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            ),
            'random_forest': lambda: RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            ),
            'svm': lambda: SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                random_state=kwargs.get('random_state', 42),
            ),
            'xgboost': lambda: xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42),
                tree_method='hist',
                device=device,
                eval_metric='logloss',
            ),
        }
    else:
        models = {
            'mlp': lambda: TorchMLPRegressor(
                hidden_dim=kwargs.get('hidden_dim', 256),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42),
                early_stopping=kwargs.get('early_stopping', True),
                validation_fraction=kwargs.get('validation_fraction', 0.1),
                use_gpu=use_gpu,
            ),
            'linear': lambda: LinearRegression(n_jobs=-1),
            'ridge': lambda: Ridge(
                alpha=kwargs.get('alpha', 1.0),
                random_state=kwargs.get('random_state', 42),
            ),
            'lasso': lambda: Lasso(
                alpha=kwargs.get('alpha', 1.0),
                random_state=kwargs.get('random_state', 42),
            ),
            'random_forest': lambda: RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1,
            ),
            'svm': lambda: SVR(kernel=kwargs.get('kernel', 'rbf')),
            'xgboost': lambda: xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                random_state=kwargs.get('random_state', 42),
                tree_method='hist',
                device=device,
            ),
        }

    if name not in models:
        available = list(models.keys())
        raise ValueError(f"Unknown head '{name}' for task '{task}'. Available: {available}")

    return models[name]()
