import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentModel(nn.Module):

    def __init__(
        self,
        seq_dim,
        path_dim,
        hidden_dim,
        use_align=True,
        use_cross=True,
    ):
        super().__init__()

        self.use_align = use_align
        self.use_cross = use_cross
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim

        # =====================
        # learnable loss weights
        # =====================
        self.log_lambda_align = nn.Parameter(torch.tensor(0.0))
        self.log_alpha_cross = nn.Parameter(torch.tensor(0.0))

        # =====================
        # Encoders（稍微加深）
        # =====================
        self.pep_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.path_proj = nn.Sequential(
            nn.Linear(path_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # =====================
        # 🔥 FiLM调制（关键新增）
        # =====================
        self.film = nn.Linear(hidden_dim, hidden_dim * 2)

        # =====================
        # 🔥 Interaction MLP（核心升级）
        # =====================
        self.mlp_reg = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, 1),
        )

        self.scale = nn.Parameter(torch.tensor(10.0))
        self.mse = nn.MSELoss()

    # =====================
    # forward
    # =====================
    def forward(self, x, mic=None, path_id=None, return_repr=False):

        pep_feat = x[:, :self.seq_dim]
        path_feat = x[:, self.seq_dim:]

        # ===== 编码 =====
        h_pep = self.pep_proj(pep_feat)
        h_path = self.path_proj(path_feat)

        # ===== normalize =====
        h_pep = F.normalize(h_pep, dim=-1)
        h_path = F.normalize(h_path, dim=-1)

        # =====================
        # 🔥 FiLM调制（替代attention）
        # =====================
        gamma, beta = self.film(h_path).chunk(2, dim=-1)
        h_pep_mod = gamma * h_pep + beta
        h_pep_mod = F.normalize(h_pep_mod, dim=-1)

        # =====================
        # 🔥 Interaction features（关键）
        # =====================
        interaction = torch.cat([
            h_pep_mod,
            h_path,
            h_pep_mod * h_path,
            torch.abs(h_pep_mod - h_path)
        ], dim=-1)

        pred = self.mlp_reg(interaction).squeeze(-1)

        if mic is None or path_id is None:
            return pred

        # =====================
        # 主任务：MSE
        # =====================
        loss_mse = self.mse(pred, mic)
        total_loss = loss_mse
        loss_dict = {"mse": loss_mse.detach()}

        # =====================
        # （可选）ranking loss
        # 👉 强烈建议你先关掉做对照实验
        # =====================
        sim = self.scale * torch.sum(h_pep_mod * h_path, dim=-1)
        mic_bucket = self.bucket_mic(mic)

        if self.use_align:
            loss_same = torch.zeros((), device=mic.device)
            count = 0
            for pid in torch.unique(path_id):
                idx = (path_id == pid).nonzero(as_tuple=True)[0]
                if idx.numel() < 2: continue
                diff_label = (mic_bucket[idx].unsqueeze(0) - mic_bucket[idx].unsqueeze(1)).float()
                diff_sim = sim[idx].unsqueeze(0) - sim[idx].unsqueeze(1)
                loss_same += F.mse_loss(diff_sim, diff_label)
                count += 1
            if count > 0:
                loss_same /= count
                lambda_align = torch.exp(self.log_lambda_align)
                total_loss += lambda_align * loss_same

        if self.use_cross:
            loss_cross = torch.zeros((), device=mic.device)
            B = mic.size(0)
            if B >= 4:
                num_pairs = min(32, B * (B - 1) // 2)
                i = torch.randint(0, B, (num_pairs,), device=mic.device)
                j = torch.randint(0, B, (num_pairs,), device=mic.device)
                mask = path_id[i] != path_id[j]
                if mask.any():
                    diff_label = (mic_bucket[i[mask]] - mic_bucket[j[mask]]).float()
                    diff_sim = sim[i[mask]] - sim[j[mask]]
                    loss_cross = F.mse_loss(diff_sim, diff_label)

                alpha_cross = torch.exp(self.log_alpha_cross)
                total_loss += alpha_cross * loss_cross

        return pred, total_loss, loss_dict

    def bucket_mic(self, mic):
        q1 = mic.quantile(0.33)
        q2 = mic.quantile(0.66)
        boundaries = torch.stack([q1, q2])
        return torch.bucketize(mic, boundaries)
