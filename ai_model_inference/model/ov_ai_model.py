# MLP Model Class Definition with Tanh Activation Function
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class ESFEncoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ESFEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: [B, 1, 10, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 5, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 2, 16]
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128 * 2 * 16, 128)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # [B, 128]


class ExtraFeatureEncoder(nn.Module):
    def __init__(self,dropout_rate=0.3):
        super(ExtraFeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(40, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.mlp(x)  # [B, 32]





class SiameseESFClassNet(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(SiameseESFClassNet, self).__init__()
        self.encoder = ESFEncoder(dropout_rate=dropout_rate)
        self.extra_encoder = ExtraFeatureEncoder()
        self.num_classes = num_classes
        self.input_size = 160

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate/2),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, esf_ref, esf_scan, extra_features):
        emb_ref = self.encoder(esf_ref)       # [B, 128]
        emb_scan = self.encoder(esf_scan)     # [B, 128]

        emb_diff = torch.abs(emb_ref - emb_scan) # [B, 128]
        emb_mult = emb_ref * emb_scan
        emb_extra = self.extra_encoder(extra_features)  # [B, 32]

        #x = torch.cat([emb_diff, emb_mult, emb_extra], dim=1)  # [B, 288]
        #x = emb_diff # 128
        x = torch.cat([emb_diff, emb_extra], dim=1)     # 128 + 32 = 160
        #x = torch.cat([emb_mult, emb_extra], dim=1)     # 128 + 32
        #x = torch.cat([emb_ref, emb_scan, emb_extra], dim=1)  # 128 + 128 + 32 = 288

        out = self.classifier(x)  # [B, 3]
        return out


class SiameseESFClassNet_Only_Xfeats(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(SiameseESFClassNet_Only_Xfeats, self).__init__()
        self.encoder = ESFEncoder(dropout_rate=dropout_rate)
        self.extra_encoder = ExtraFeatureEncoder()
        self.num_classes = num_classes
        self.input_size = 40#160

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate/2),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, esf_ref, esf_scan, extra_features):
        #emb_ref = self.encoder(esf_ref)       # [B, 128]
        #emb_scan = self.encoder(esf_scan)     # [B, 128]

        #emb_diff = torch.abs(emb_ref - emb_scan) # [B, 128]
        #emb_mult = emb_ref * emb_scan
        #emb_extra = self.extra_encoder(extra_features)  # [B, 32]

        #x = torch.cat([emb_diff, emb_mult, emb_extra], dim=1)  # [B, 288]
        #x = emb_diff # 128
        #x = torch.cat([emb_diff, emb_extra], dim=1)     # 128 + 32 = 160
        #x = torch.cat([emb_mult, emb_extra], dim=1)     # 128 + 32
        #x = torch.cat([emb_ref, emb_scan, emb_extra], dim=1)  # 128 + 128 + 32 = 288
        x = extra_features
        out = self.classifier(x)  # [B, 3]
        return out


class ESFResNetCNN_withVector_old(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()

        # CNN Backbone
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = ResidualBlock(32, 64, downsample=True)
        self.block2 = ResidualBlock(64, 128, downsample=True)
        self.block3 = ResidualBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d((2,4))
        self.flatten_dim = 128 * 2 * 4

        # CNN MLP Head
        self.cnn_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Extra-Feature MLP (für 40-D Vektor)
        self.extra_mlp = nn.Sequential(
            nn.Linear(40, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Final classifier
        self.fc_out = nn.Linear(64 + 32, num_classes)

    def forward(self, x, extra_feats):
        # CNN Pfad
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        cnn_feat = self.cnn_mlp(x)  # (B, 64)

        # Extra-Features Pfad
        extra_out = self.extra_mlp(extra_feats)  # (B, 32)

        # Kombinieren
        combined = torch.cat([cnn_feat, extra_out], dim=1)  # (B, 96)
        out = self.fc_out(combined)
        return out


# Verbesserte Hauptarchitektur
class ESFResNetCNN_withVector(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()

        # 🔹 CNN Backbone
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # weniger Kanäle
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.block1 = ResidualBlock(16, 32, downsample=True)
        self.block2 = ResidualBlock(32, 64, downsample=True)
        self.block3 = ResidualBlock(64, 64)
        self.block4 = ResidualBlock(64, 128, downsample=True)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.flatten_dim = 64 * 2 * 4

        # 🔹 CNN MLP Head
        self.cnn_mlp = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 🔹 Extra-Feature MLP (für 40-D Vektor)
        self.extra_mlp = nn.Sequential(
            nn.Linear(40, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # 🔹 Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 🔹 Final classifier
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x, extra_feats):
        # CNN Pfad
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        cnn_feat = self.cnn_mlp(x)  # (B, 64)

        # Extra-Features Pfad
        extra_out = self.extra_mlp(extra_feats)  # (B, 16)

        # Fusion
        combined = torch.cat([cnn_feat, extra_out], dim=1)  # (B, 80)
        fused = self.fusion_mlp(combined)  # (B, 64)

        out = self.fc_out(fused)  # (B, num_classes)
        return out



class SiameseESFClassNet_NO_X(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(SiameseESFClassNet_NO_X, self).__init__()
        self.encoder = ESFEncoder(dropout_rate=dropout_rate)
        #self.extra_encoder = ExtraFeatureEncoder()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate/2),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, esf_ref, esf_scan, extra_features):
        emb_ref = self.encoder(esf_ref)       # [B, 128]
        emb_scan = self.encoder(esf_scan)     # [B, 128]

        emb_diff = torch.abs(emb_ref - emb_scan) # [B, 128]
        emb_mult = emb_ref * emb_scan
        #emb_extra = self.extra_encoder(extra_features)  # [B, 32]

        #x = torch.cat([emb_diff, emb_mult, emb_extra], dim=1)  # [B, 288]
        x = emb_diff # 128
        #x = torch.cat([emb_diff, emb_extra], dim=1)     # 128 + 32 = 160
        #x = torch.cat([emb_mult, emb_extra], dim=1)     # 128 + 32
        #x = torch.cat([emb_ref, emb_scan, emb_extra], dim=1)  # 128 + 128 + 32 = 288

        out = self.classifier(x)  # [B, 3]
        return out

class ESFEncoder_S(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ESFEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 5, 32]

            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 96, 2, 16]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(96 * 2 * 16, 64)
        )

    def forward(self, x):
        return self.fc(self.conv(x))  # [B, 64]
class ExtraFeatureEncoder_S(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(40, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.mlp(x)  # [B, 16]
class SiameseESFClassNet_S(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.encoder = ESFEncoder(dropout_rate)
        self.extra_encoder = ExtraFeatureEncoder(dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, esf_ref, esf_scan, extra_features):
        emb_ref = self.encoder(esf_ref)  # [B, 64]
        emb_scan = self.encoder(esf_scan)  # [B, 64]

        emb_diff = torch.abs(emb_ref - emb_scan)  # [B, 64]
        emb_extra = self.extra_encoder(extra_features)  # [B, 16]

        x = torch.cat([emb_diff, emb_extra], dim=1)  # [B, 80]
        return self.classifier(x)

class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Initialisiert die Focal Loss.

        Args:
        - alpha (float): Gewichtung für das Ungleichgewicht zwischen Klassen (Standard: 0.25).
        - gamma (float): Fokussierungsfaktor für schwer zu klassifizierende Beispiele (Standard: 2).
        - reduction (str): Aggregationsmethode für den Loss ('mean', 'sum', 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Berechnet den Focal Loss.

        Args:
        - logits (Tensor): Modell-Logits der Form [BatchSize, NumClasses].
        - targets (Tensor): Ziel-Labels der Form [BatchSize] (als Integer-Klassen-Indices).

        Returns:
        - Tensor: Der berechnete Focal Loss.
        """
        # Softmax auf Logits anwenden, um Wahrscheinlichkeiten zu erhalten
        probs = F.softmax(logits, dim=1)

        # Wahrscheinlichkeiten für die Zielklassen extrahieren
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))
        probs_for_targets = torch.sum(probs * targets_one_hot, dim=1)

        # Berechnung der modifizierten Cross-Entropy
        log_probs = torch.log(probs_for_targets)
        focal_weight = (1 - probs_for_targets) ** self.gamma

        loss = -self.alpha * focal_weight * log_probs

        # Reduktion anwenden
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class FocalLoss_backup(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        # Reduktion anwenden
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha  # float oder None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # [B]

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]  # Klassengewicht je Sample
        else:
            at = self.alpha if self.alpha is not None else 1.0

        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DualMetricScheduler:
    def __init__(self, optimizer, patience, mode="max"):
        self.optimizer = optimizer
        self.patience = patience
        self.mode = mode
        self.best_loss = float("inf")
        self.best_f1 = 0
        self.wait = 0
        self.min_lr = 1e-6  # als init-Argument oder fix

    def step(self, val_loss, val_f1):
        if self.mode == "max":
            if val_f1 > self.best_f1 or val_loss < self.best_loss:
                self.best_f1 = max(self.best_f1, val_f1)
                self.best_loss = min(self.best_loss, val_loss)
                self.wait = 0
            else:
                self.wait += 1
        if self.wait >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * 0.1, self.min_lr)
                if new_lr < param_group['lr']:
                    print(f"📉 Reducing LR to {new_lr}")
                param_group['lr'] = new_lr
                print(f"📉 Reducing LR to {param_group['lr']} due to no improvement.")
            self.wait = 0

class SoftF1Loss(nn.Module):
    def __init__(self):
        super(SoftF1Loss, self).__init__()

    def forward(self, outputs, targets):
        probs = torch.sigmoid(outputs)  # Für Multi-Label oder Binary
        tp = (probs * targets).sum(dim=0)
        fp = ((1 - targets) * probs).sum(dim=0)
        fn = (targets * (1 - probs)).sum(dim=0)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        return 1 - f1.mean()  # 1 - F1-Score




class ResidualBlock_back(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), downsample=False):
        super(ResidualBlock, self).__init__()
        stride = (2, 2) if downsample else (1, 1)
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return F.relu(out)


class ESFResNetCNN_back(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(ESFResNetCNN, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock(32, 64, downsample=True)
        self.block2 = ResidualBlock(64, 128, downsample=True)
        self.block3 = ResidualBlock(128, 128)

        self.pool = nn.AdaptiveAvgPool2d((2, 4))  # → (B, 128, 2, 4)
        self.fc1 = nn.Linear(128 * 2 * 4, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), downsample=False):
        super(ResidualBlock, self).__init__()
        stride = (2, 2) if downsample else (1, 1)
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)
        self.residual = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return F.relu(out)

class ESFResNetCNN(nn.Module):
    def __init__(self, num_classes = 2, dropout_rate=0.3):
        super(ESFResNetCNN, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock(32, 64, downsample=True)
        self.block2 = ResidualBlock(64, 128, downsample=True)
        self.block3 = ResidualBlock(128, 128)

        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc1 = nn.Linear(128 * 2 * 4, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)  # 1 Output für binär

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # shape: [B,1]
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------ #
# 0)  Hilfs-Module
# ------------------------------------------------ #
class DepthwiseStem(nn.Module):
    """3×3 depthwise + 1×1 pointwise → weniger FLOPs"""
    def __init__(self, in_ch=4, out_ch=32):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn    = nn.GroupNorm(8, out_ch)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return F.silu(self.gn(x))

# ---------- Channel & Spatial Attention (CBAM) ---------- #
class ChannelGate(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)
    def forward(self, x):
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        m = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        m = torch.sigmoid(self.conv(m))
        return x * m

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel = ChannelGate(ch)
        self.spatial = SpatialGate()
    def forward(self, x):
        return self.spatial(self.channel(x))

# ---------- Residual-Block + CBAM ---------- #
class ResidualBlock2(nn.Module):
    def __init__(self, inp, outp, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(inp, outp, 3, stride, 1, bias=False)
        self.gn1   = nn.GroupNorm(8, outp)
        self.conv2 = nn.Conv2d(outp, outp, 3, 1, 1, bias=False)
        self.gn2   = nn.GroupNorm(8, outp)
        self.att   = CBAM(outp)
        self.skip  = nn.Identity() if (inp == outp and not down) else nn.Sequential(
            nn.Conv2d(inp, outp, 1, stride, bias=False),
            nn.GroupNorm(8, outp))
    def forward(self, x):
        out = F.silu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.att(out)
        out = out + self.skip(x)
        return F.silu(out)

# ---------- DropBlock (very light) ---------- #
class DropBlock2D(nn.Module):
    def __init__(self, p=0.1, block_size=3):
        super().__init__()
        self.p = float(p)
        self.bs = int(block_size)

    def forward(self, x):
        if (not self.training) or self.p <= 0.0:
            return x

        b, c, h, w = x.shape
        if self.bs <= 1 or self.bs > min(h, w):
            return x  # nichts zu maskieren

        # Gamma nach Paper
        gamma = self.p * (h * w) / (self.bs ** 2) / ((h - self.bs + 1) * (w - self.bs + 1))

        # Seed-Maske (kleiner)
        seed = (torch.rand(b, 1, h - self.bs + 1, w - self.bs + 1, device=x.device) < gamma).float()

        # Auf volle Größe bringen: erst paddden, dann max_pool2d ANWENDEN (nicht als Methode)
        pad = (self.bs // 2, self.bs // 2, self.bs // 2, self.bs // 2)  # (l, r, t, b)
        seed = F.pad(seed, pad)
        block_mask = F.max_pool2d(seed, kernel_size=self.bs, stride=1, padding=self.bs // 2)

        keep = 1.0 - block_mask
        # Re-skalieren, damit der Erwartungswert erhalten bleibt
        denom = keep.sum().clamp(min=1.0)
        keep = keep * (keep.numel() / denom)

        return x * keep
# ------------------------------------------------ #
#  Main Model
# ------------------------------------------------ #
class ESFResNetCNN_withVector_Att(nn.Module):
    def __init__(self, dropout_rate_cnn=0.15, dropout_rate_x=0.15,
                 dropout_rate_fuse=0.05, num_classes=2, extra_dim=44):
        super().__init__()
        self.stem = DepthwiseStem(4, 32)
        self.res1 = ResidualBlock2(32, 32)
        self.res2 = ResidualBlock2(32, 64, down=True)
        self.dropblock = DropBlock2D(p=0.08)   # vorher 0.10
        self.res3 = ResidualBlock2(64, 64)
        self.res4 = ResidualBlock2(64, 128, down=True)

        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.flat = 128 * 2 * 4

        self.cnn_head = nn.Sequential(
            nn.Linear(self.flat, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate_cnn),   # 0.15
            nn.Linear(128, 64),
            nn.GELU()
        )

        self.x_head = nn.Sequential(
            nn.Linear(extra_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate_x),     # 0.15
            nn.Linear(32, 16),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 16, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate_fuse)   # 0.05 (oder 0.0)
        )

        self.fc_out = nn.Linear(64, num_classes)


    # --------- HIER: forward ----------
    def forward(self, img, feats):
        """
        img   : Tensor (B, 4, 11, 64)
        feats : Tensor (B, extra_dim)
        Rückgabe: Logits (B, 2)
        """
        assert feats.dim() == 2, f"extra features müssen 2D sein, got {tuple(feats.shape)}"

        x = self.stem(img)
        x = self.res1(x)
        x = self.dropblock(self.res2(x))
        x = self.res3(x)
        x = self.res4(x)

        x = self.pool(x).flatten(1)      # (B, self.flat)
        cnn_vec  = self.cnn_head(x)      # (B, 64)
        feat_vec = self.x_head(feats)    # (B, 16)

        fused = self.fusion(torch.cat([cnn_vec, feat_vec], dim=1))  # (B, 64)
        return self.fc_out(fused)        # (B, 2) Logits für CrossEntropy
# new dataset models

# PyTorch-Skizzen
class DropPath(nn.Module):
    def __init__(self, p=0.02):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, device=x.device).bernoulli_(keep)
        return x * mask / keep


# --- Block-SE: Gating über Block-Achse (11 Blöcke à 64) ---
class BlockSE(nn.Module):
    """
    Gated weighting über Blöcke: x_main -> (B, 704), in 11 Blöcke à 64 aufteilen,
    je Block zu einem Skalar poolen, kleines MLP über 11er-Vektor, Sigmoid als Gate,
    Gate zurück auf Blöcke broadcasten.
    """
    def __init__(self, num_blocks=11, reduction=4, block_size=64):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        hid = max(1, self.num_blocks // int(reduction))
        self.ln_blocks = nn.LayerNorm(self.block_size)   # pro Block normieren (stabiler)
        self.fc1 = nn.Linear(self.num_blocks, hid)
        self.fc2 = nn.Linear(hid, self.num_blocks)

    def forward(self, x_main):
        # x_main: (B, D= num_blocks*block_size)
        B, D = x_main.shape
        assert D % self.block_size == 0, "block_size passt nicht zu D"
        nb = D // self.block_size
        assert nb == self.num_blocks, f"erwartet {self.num_blocks} Blöcke, gefunden {nb}"

        xb = x_main.view(B, nb, self.block_size)            # (B, 11, 64)
        xb = self.ln_blocks(xb)                              # LN pro Block
        s  = xb.abs().mean(dim=-1)                          # (B, 11) Block-Score via MEAN(|x|)
        g  = torch.sigmoid(self.fc2(F.gelu(self.fc1(s))))   # (B, 11) Gate je Block
        #xb = xb * g.unsqueeze(-1)                          #(B, 11, 64) anwenden
        alpha = 0.5  # 0.5…1.0; 0.5 ist sicher
        g_res = 1.0 + alpha * (g - 0.5)  # Range ~ [0.75, 1.25] bei alpha=0.5
        xb = xb * g_res.unsqueeze(-1)
        return xb.reshape(B, D)

class ResidualMLP(nn.Module):
    def __init__(self, d, hidden, p_hidden=0.1, droppath=0.05):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.act = nn.GELU()
        self.drop_hidden = nn.Dropout(p_hidden)
        self.drop_path = DropPath(droppath)
    def forward(self, x):
        h = self.ln(x)
        h = self.drop_hidden(self.act(self.fc1(h)))
        h = self.fc2(h)
        return x + self.drop_path(h)

class SE1D(nn.Module):
    def __init__(self, d, r=16):
        super().__init__()
        h = max(d // r, 1)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
    def forward(self, x):                      # x: (B, d)
        g = torch.sigmoid(self.fc2(F.gelu(self.fc1(x))))
        return x * g

class PreGate(nn.Module):
    def __init__(self, d_in, d_att=32, gate_ratio=4):
        super().__init__()
        self.proj_down = nn.Linear(d_in, d_att)
        self.gate      = SE1D(d_att, r=gate_ratio)
        self.proj_up   = nn.Linear(d_att, d_in)
    def forward(self, x):
        z = self.proj_down(x)
        z = self.gate(z)
        g = torch.sigmoid(self.proj_up(z))
        # statt: return x * g
        alpha = 0.5
        g_res = 1.0 + alpha * (g - 0.5)
        return x * g_res

class ExtraHead(nn.Module):
    def __init__(self, d_in, d_out=64, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Dropout(p)
        )
    def forward(self, x):
        return self.net(x)

class MLP704(nn.Module):
    def __init__(self, in_main=704, in_extra=44, out_dim=2,
                 width=512, depth=4, p=0.3, se_ratio=16,
                 use_block_att=True, block_reduction=4, block_size=64):
        super().__init__()
        self.in_main = int(in_main)
        self.in_extra = int(in_extra)

        # (optional) Block-Attention/Gating NUR auf den 704er Hauptvektor
        self.block_att = BlockSE(num_blocks=self.in_main // block_size,
                                 reduction=block_reduction,
                                 block_size=block_size) if use_block_att else None

        d_in = in_main + in_extra
        self.pregate = PreGate(d_in=d_in, d_att=32, gate_ratio=4)

        self.embed = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, width),
            nn.GELU(),
        )

        # DropPath über Tiefe leicht ansteigend (optional)
        self.blocks = nn.ModuleList([
            ResidualMLP(width, width*2, p_hidden=p, droppath=(0.05 + i*(0.10-0.05)/max(1, depth-1)))
            for i in range(depth)
        ])

        self.se = SE1D(width, r=se_ratio)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_main, x_extra=None):
        # 1) Optionales Block-Gating auf dem 704er Hauptvektor
        if (self.block_att is not None) and (x_main is not None):
            x_main = self.block_att(x_main)     # (B, 704)

        # 2) Extras anhängen (wenn vorhanden)
        if (x_extra is None) or (x_extra.numel() == 0):
            x = x_main
        else:
            x = torch.cat([x_main, x_extra], dim=1)   # (B, 704+in_extra)

        # 3) PreGate + MLP-Backbone
        x = self.pregate(x)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.se(x)
        return self.head(x)


class MLP704v2(nn.Module):
    def __init__(self, in_main=704, d_esf_norm=44, d_grid=27, out_dim=2,
                 width=512, depth=4, p=0.3, se_ratio=16,
                 use_block_att=True, block_reduction=4, block_size=64):
        super().__init__()
        self.in_main = int(in_main)
        self.d_esf_norm = int(d_esf_norm)
        self.d_grid = int(d_grid)

        self.block_att = BlockSE(num_blocks=self.in_main // block_size,
                                 reduction=block_reduction,
                                 block_size=block_size) if use_block_att else None

        # separate, struktur-bewusste Einbettung der Nebenkanäle
        self.head_esf_norm = ExtraHead(self.d_esf_norm, d_out=64)   # 44 → 64
        self.head_grid     = ExtraHead(self.d_grid,     d_out=64)   # 27 → 64

        d_in_fused = in_main + 64 + 64
        self.pregate = PreGate(d_in=d_in_fused, d_att=32, gate_ratio=4)

        self.embed = nn.Sequential(
            nn.LayerNorm(d_in_fused),
            nn.Linear(d_in_fused, width),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResidualMLP(width, width*2, p_hidden=p, droppath=(0.05 + i*(0.10-0.05)/max(1, depth-1)))
            for i in range(depth)
        ])
        self.se = SE1D(width, r=se_ratio)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, out_dim),
        )

    def forward(self, x_main, x_esf_norm=None, x_grid=None):
        # 704er Hauptvektor mit BlockSE
        if (self.block_att is not None) and (x_main is not None):
            x_main = self.block_att(x_main)

        # Nebenarme robust einbetten (fehlen erlaubt → Nullen)
        if x_esf_norm is None:
            x_esf_norm = x_main.new_zeros(x_main.size(0), self.d_esf_norm)
        if x_grid is None:
            x_grid     = x_main.new_zeros(x_main.size(0), self.d_grid)

        z_esf_norm = self.head_esf_norm(x_esf_norm)
        z_grid     = self.head_grid(x_grid)

        x = torch.cat([x_main, z_esf_norm, z_grid], dim=1)

        x = self.pregate(x)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.se(x)
        return self.head(x)

class MLPFlex(nn.Module):
    """
    Modi:
      - alle:   use_main=True,  use_esf_norm=True,  use_grid=True
      - grid:   use_main=False, use_esf_norm=False, use_grid=True
      - extra:  use_main=False, use_esf_norm=True,  use_grid=False
      - main:   use_main=True,  use_esf_norm=False, use_grid=False
    """
    def __init__(self,
                 in_main=704, d_esf_norm=44, d_grid=27, out_dim=2,
                 width=512, depth=4, p=0.3, se_ratio=16,
                 use_block_att=True, block_reduction=4, block_size=64,
                 use_main=True, use_esf_norm=True, use_grid=True):
        super().__init__()

        # Flags
        self.use_main     = bool(use_main)
        self.use_esf_norm = bool(use_esf_norm)
        self.use_grid     = bool(use_grid)

        # Dims
        self.in_main   = int(in_main)
        self.d_esf_norm= int(d_esf_norm)
        self.d_grid    = int(d_grid)

        # === Main-Kanal ===
        self.block_att = None
        self.ln_main   = None
        if self.use_main:
            if use_block_att:
                # Anzahl Blöcke aus Dim ableiten
                num_blocks = self.in_main // block_size
                assert num_blocks * block_size == self.in_main, "block_size teilt in_main nicht."
                self.block_att = BlockSE(num_blocks=num_blocks,
                                         reduction=block_reduction,
                                         block_size=block_size)
            self.ln_main = nn.LayerNorm(self.in_main)

        # === Nebenkanäle ===
        self.head_esf_norm = ExtraHead(self.d_esf_norm, d_out=64, p=p) if self.use_esf_norm else None
        self.head_grid     = ExtraHead(self.d_grid,     d_out=64, p=p) if self.use_grid     else None

        # === Fusionsdimension bestimmen ===
        d_in_fused = 0
        if self.use_main:     d_in_fused += self.in_main
        if self.use_esf_norm: d_in_fused += 64
        if self.use_grid:     d_in_fused += 64
        assert d_in_fused > 0, "Mindestens ein Kanal muss aktiv sein."

        # === Pregate + Embed ===
        self.pregate = PreGate(d_in=d_in_fused, d_att=32, gate_ratio=4)
        self.embed = nn.Sequential(
            nn.LayerNorm(d_in_fused),
            nn.Linear(d_in_fused, width),
            nn.GELU(),
        )

        # === Backbone ===
        self.blocks = nn.ModuleList([
            ResidualMLP(width, width*2, p_hidden=p, droppath=(0.05 + i*(0.10-0.05)/max(1, depth-1)))
            for i in range(depth)
        ])
        self.se = SE1D(width, r=se_ratio)

        # === Kopf ===
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _infer_B_and_device(*xs):
        dev = None
        B = None
        for t in xs:
            if t is not None:
                B = t.size(0)
                dev = t.device
                break
        if B is None:
            raise ValueError("Alle Eingaben sind None. Batchgröße unbekannt.")
        return B, dev

    def forward(self, x_main=None, x_esf_norm=None, x_grid=None):
        B, dev = self._infer_B_and_device(x_main, x_esf_norm, x_grid)
        parts = []

        # --- main ---
        if self.use_main:
            if x_main is None:
                x_main = torch.zeros(B, self.in_main, device=dev)
            if self.block_att is not None:
                x_main = self.block_att(x_main)      # (B, 704)
            x_main = self.ln_main(x_main)            # stabilisiert
            parts.append(x_main)

        # --- esf/extra (44) ---
        if self.use_esf_norm:
            if x_esf_norm is None:
                x_esf_norm = torch.zeros(B, self.d_esf_norm, device=dev)
            z_esf = self.head_esf_norm(x_esf_norm)   # (B, 64)
            parts.append(z_esf)

        # --- grid (27) ---
        if self.use_grid:
            if x_grid is None:
                x_grid = torch.zeros(B, self.d_grid, device=dev)
            z_grid = self.head_grid(x_grid)          # (B, 64)
            parts.append(z_grid)

        # --- Fuse ---
        x = torch.cat(parts, dim=1)

        # --- Pregate + MLP ---
        x = self.pregate(x)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.se(x)
        return self.head(x)

