import torch
from model.Module.Utils import *

def get_device(use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        return 'cuda'
    elif use_gpu and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class MyLSTM(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 layer_num,
                 hidden_dim_fc1,
                 hidden_dim_fc2,
                 # hidden_dim_fc3,
                 bidirecion=False,
                 dropout_lstm=0,
                 dropout_fc=0):
        super(MyLSTM, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        # self._embed_dim=embed_dim #embed_dim=None,
        self.hidden_dim_lstm=hidden_dim
        self.layer_num_lstm=layer_num
        self.hidden_dim_fc1=hidden_dim_fc1
        self.hidden_dim_fc2=hidden_dim_fc2
        # self._hidden_dim_fc3=hidden_dim_fc3
        #
        self.dropout_lstm=dropout_lstm
        self.dropout_fc=dropout_fc
        self.bidirecion=bidirecion
        #
        # self.embedding_layers=torch.nn.Embedding(num_embeddings=self._input_dim,embedding_dim=self._embed_dim) 'embedding_layers, input must be tensor-int'
        self.lstm_layers=torch.nn.LSTM(
            # input_size=self._embed_dim, 'embedding_layers'
            input_size=self.input_dim,
            num_layers=self.layer_num_lstm,
            hidden_size=self.hidden_dim_lstm,
            bias=True,
            bidirectional=self.bidirecion,
            batch_first=True,
            dropout=self.dropout_lstm
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim_lstm, self.hidden_dim_fc1),
            # torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(p=self.dropout_fc),
            torch.nn.Linear(self.hidden_dim_fc1, self.hidden_dim_fc2),
            # torch.nn.LeakyReLU(0.01),
            # torch.nn.Dropout(p=self._dropout_fc),
            # torch.nn.Linear(self._hidden_dim_fc2, self._hidden_dim_fc3),
            # torch.nn.Dropout(p=self._dropout_fc),
            torch.nn.Linear(self.hidden_dim_fc2, self.output_dim),
            # torch.nn.ReLU()
        )

    def forward(self, x, h_state_initial = None):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            out (Tensor): Regression prediction of shape [batch_size, output_dim].
            features (Tensor): Feature representation from the LSTM of shape [batch_size, hidden_dim].
        """
        # lstm_out: [batch_size, seq_len, hidden_dim]
        # h_n: [num_layers, batch_size, hidden_dim]
        lstm_out, (h_n, _) = self.lstm_layers(x, h_state_initial)
        # Use the final layer's last hidden state as the feature representation
        features = h_n[-1]  # shape: [batch_size, hidden_dim]
        out_fc = self.fc(lstm_out)
        return out_fc, features

def calculate_ec_loss(T_s, phys):
    """
    Calculate energy-driven loss per batch
    based on the ODE residual:
    Residual = (c*rho*d_z)*dT_s/dt - F

    Args:
        T_s (batch_size, seq_len,1): Labels predicted by LSTM
        phys (batch_size, seq_len, n_vars): Physical variables:['SWdown', 'LWdown', 'AirTemp', 'RelHum', 'WindSpeed', 'Ice']

    Returns:
        Average energy-driven loss per batch (W/m2)
        --> tensor (1)
    """
    # T_s = y_pred
    # Loop through the batch
    T_s = T_s[:,:,0]
    batch_size = T_s.shape[0]
    loss_ec_per_batch = []
    for i in range(batch_size):
        T_s_batch = T_s[i,:]
        phy_batch = phys[i, :, :]
        # Calculate thermal storage energy change at each timestep (dQ/dt)
        Q_t = calculate_thermal_storage_change_rate(T_s_batch)
        # Calculate the net heat flux over the same timestep  (F)
        F = calculate_net_heat_fluxes(phy_batch, T_s_batch)
        # Calculate energy-driven loss at each timestep
        loss_ec_per_timestep = torch.abs(Q_t - F)
        # Record energy-driven loss per batch
        loss_ec_per_batch.append(loss_ec_per_timestep)
    loss_ec = torch.concat(loss_ec_per_batch) # list --> tensor
    # relu to remove negative values
    loss_ec = torch.clamp(loss_ec, min=0.0)
    return loss_ec.mean()

def mmd_loss(src_features, tgt_features, kernel_fn=rbf_kernel):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between two distributions.
    source code see: https://github.com/ZongxianLee/MMD_Loss.Pytorch.git & https://github.com/yiftachbeer/mmd_loss_pytorch.git
    Args:
        src_features (Tensor): set of source features, shape [n_samples_X, feature_dim].
        tgt_features (Tensor): set of target features, shape [n_samples_X, feature_dim].
    Returns:
        Tensor: A scalar representing the MMD loss. Lower values indicate more similar distributions.
    """

    combined = torch.vstack([src_features, tgt_features])
    K = kernel_fn(combined)

    X_size = src_features.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()

    return XX - 2 * XY + YY


class HybridLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(HybridLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y, phys, lam_ec=0.001, phycis_informed=True):
        """
        Hybrid loss that combines data regression loss and energy conservation loss.

        Args:
            y_pred (Tensor): Predictions.
            y (Tensor): Ground truth labels.
            phys (Tensor): Physical constraints input.
            lam_ec (float): Weight for the energy conservation (EC) loss term.
            physics_informed (bool): Flag to indicate if EC loss should be computed.

        Returns:
            Tuple (loss, loss_data, loss_ec)
        """
        # Create mask for valid targets: non-NaN and strictly greater than 0.
        mask = (~torch.isnan(y)) & (y > 0)
        # mask = mask.squeeze()  # Remove extra dimensions if any.
        y_pred_valid = y_pred[mask]
        y_valid = y[mask]
        # valid data loss
        loss_data = torch.nn.functional.mse_loss(y_pred_valid, y_valid, reduction='mean')
        # Avoid NaN loss (if valid set is empty, for example).
        if torch.isnan(loss_data):
            loss_data = torch.zeros(1, device=y.device, dtype=y.dtype,requires_grad=True)
        # Initialize energy conservation (EC) loss to zero.
        loss_ec = torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype,requires_grad=True)
        if phycis_informed:
            loss_ec = calculate_ec_loss(y_pred, phys)
            # If the EC loss is NaN, reset to zero.
            if torch.isnan(loss_ec):
                loss_ec = torch.zeros(1, device=y_pred.device, dtype=y_pred.dtype, requires_grad=True)
        # Total loss is the sum of the data loss and weighted EC loss.
        loss = loss_data + lam_ec * loss_ec
        return loss, loss_data, loss_ec


class Domain_adaptaion_Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(Domain_adaptaion_Loss, self).__init__()
        self.reduction = reduction

    def forward(self,
                src_pred, src_labels, src_features,
                mid_pred, mid_labels, mid_features,
                lam_src=1, lam_mid=0.5, lam_mmd=0.1):
        """
                Calculate the domain adaptation loss for multi-stage transfer learning.
                The loss is a weighted sum of:
                    - Source regression loss (MSE), computed only on valid samples.
                    - Middle domain regression loss (MSE), computed only on valid samples.
                    - MMD loss computed on valid feature representations.

                Valid samples for regression are those where the labels and predictions
                are non-NaN and strictly greater than 0.
                For features, a sample is valid if it does not contain any NaNs and is not an entire row of zeros.

                Args:
                    src_pred (Tensor): Source domain predictions.
                    src_labels (Tensor): Source domain labels.
                    src_features (Tensor): Source domain features.
                    mid_pred (Tensor): Middle domain predictions.
                    mid_labels (Tensor): Middle domain labels.
                    mid_features (Tensor): Middle domain features.
                    lam_src (float): Weight for source regression loss.
                    lam_mid (float): Weight for middle domain regression loss.
                    lam_mmd (float): Weight for MMD loss.

                Returns:
                    tuple: (total_loss, loss_src, loss_mid, loss_mmd)
        """

        # Create a mask of valid (non-NaN) positions
        # src_labels = torch.randn(900,1)
        # mid_labels = torch.randn(90,1)
        # src_features = torch.randn(900,6)
        # mid_features = torch.randn(90,6)
        mask_src = ~torch.isnan(src_labels) & (src_labels != 0)
        mask_mid = ~torch.isnan(mid_labels) & (mid_labels != 0)
        # mask_src = mask_src.squeeze()
        # mask_mid = mask_mid.squeeze()

        # Calculate source regression loss only for valid samples
        if mask_src.sum() > 0:
            loss_src = torch.nn.functional.mse_loss(src_pred[mask_src], src_labels[mask_src], reduction='mean')
        else:
            loss_src = torch.zeros(1, device=src_pred.device, dtype=src_pred.dtype, requires_grad=True)
        # Calculate middle domain regression loss only for valid samples
        if mask_mid.sum() > 0:
            loss_mid = torch.nn.functional.mse_loss(mid_pred[mask_mid], mid_labels[mask_mid], reduction='mean')
        else:
            loss_mid = torch.zeros(1, device=mid_pred.device, dtype=mid_pred.dtype, requires_grad=True)

        # Create masks for valid features
        # Only skip samples where the entire row is all zeros.
        valid_src_feat_mask = (~torch.isnan(src_features)).all(dim=1) & (~(src_features.eq(0)).all(dim=1))
        valid_mid_feat_mask = (~torch.isnan(mid_features)).all(dim=1) & (~(mid_features.eq(0)).all(dim=1))

        # Select only valid feature vectors for MMD calculation
        src_features_valid = src_features[valid_src_feat_mask]
        mid_features_valid = mid_features[valid_mid_feat_mask]

        # Compute MMD loss only if both sets have valid samples
        if src_features_valid.shape[0] > 0 and mid_features_valid.shape[0] > 0:
            loss_mmd = mmd_loss(src_features_valid, mid_features_valid)
        else:
            loss_mmd = torch.zeros(1, device=src_features.device, dtype=src_features.dtype, requires_grad=True)
        #
        loss = lam_src*loss_src + lam_mid * loss_mid + lam_mmd * loss_mmd
        return loss, loss_src, loss_mid, loss_mmd