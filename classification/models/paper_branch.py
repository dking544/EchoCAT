import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperResidualBranch(nn.Module):


    def __init__(self,
                 in_channels=576,
                 branch_channels=192,
                 depth=2,
                 decoder_depth=2,
                 num_heads=3,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 mask_ratio=0.1,
                 patch_size=32,
                 image_channels=3,
                 num_patches=49,
                 reconstruction_weight=0.1,
                 classification_uses_masked_tokens=True):
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1).")
        if branch_channels % num_heads != 0:
            raise ValueError("branch_channels must be divisible by num_heads.")

        self.in_channels = int(in_channels)
        self.branch_channels = int(branch_channels)
        self.mask_ratio = float(mask_ratio)
        self.patch_size = int(patch_size)
        self.image_channels = int(image_channels)
        self.num_patches = int(num_patches)
        self.reconstruction_weight = float(reconstruction_weight)


        self.classification_uses_masked_tokens = bool(
            classification_uses_masked_tokens)

        self.input_projection = nn.Conv2d(in_channels, branch_channels, 1)
        self.position_projection = nn.Conv2d(
            branch_channels, branch_channels, 3, padding=1,
            groups=branch_channels)
        self.class_token = nn.Parameter(torch.zeros(1, 1, branch_channels))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, branch_channels))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, branch_channels))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_channels,
            nhead=num_heads,
            dim_feedforward=int(branch_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(branch_channels)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=branch_channels,
            nhead=num_heads,
            dim_feedforward=int(branch_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.reconstruction_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(branch_channels)
        patch_values = image_channels * patch_size * patch_size
        self.patch_prediction = nn.Linear(branch_channels, patch_values)


        self.zero_conv = nn.Conv1d(branch_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _spatial_shape(num_tokens):
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"A square token map is required, got {num_tokens} tokens.")
        return side, side

    def make_mask(self, cam):
        batch_size, num_tokens = cam.shape
        if self.mask_ratio <= 0.0:
            return torch.zeros_like(cam, dtype=torch.bool)
        num_masked = max(1, int(round(num_tokens * self.mask_ratio)))
        num_masked = min(num_masked, num_tokens - 1)
        indices = cam.topk(num_masked, dim=1, largest=True).indices
        mask = torch.zeros(
            batch_size, num_tokens, dtype=torch.bool, device=cam.device)
        mask.scatter_(1, indices, True)
        return mask

    def patchify(self, images, expected_tokens):
        patches = F.unfold(
            images, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        if patches.shape[1] != expected_tokens:
            raise ValueError(
                f"patch_size={self.patch_size} produced {patches.shape[1]} "
                f"patches, but TinyViT returned {expected_tokens} tokens.")
        return patches

    def forward(self, spatial_tokens, images, external_cam, reconstruct=False):
        batch_size, num_tokens, channels = spatial_tokens.shape
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}.")
        if num_tokens != self.num_patches:
            raise ValueError(
                f"Configured for {self.num_patches} patches, got {num_tokens}.")
        if external_cam.shape != (batch_size, num_tokens):
            raise ValueError(
                f"Expected CAM shape {(batch_size, num_tokens)}, "
                f"got {tuple(external_cam.shape)}.")

        height, width = self._spatial_shape(num_tokens)
        feature_map = spatial_tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width)
        projected = self.input_projection(feature_map)
        projected = projected + self.position_projection(projected)
        tokens = projected.flatten(2).transpose(1, 2)

        patch_positions = self.position_embedding[:, 1:num_tokens + 1]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        cam = external_cam.detach()
        mask = self.make_mask(cam) if reconstruct else torch.zeros_like(
            cam, dtype=torch.bool)
        reconstruction_loss = spatial_tokens.new_zeros(())
        classification_uses_mask = bool(
            reconstruct and mask.any()
            and self.classification_uses_masked_tokens)


        encoded_class = None
        if not classification_uses_mask:
            full_sequence = torch.cat((class_tokens, tokens), dim=1)
            full_sequence = (
                full_sequence + self.position_embedding[:, :num_tokens + 1])
            encoded_full = self.encoder_norm(self.encoder(full_sequence))
            encoded_class = encoded_full[:, 0]

        if reconstruct and mask.any():
            visible_tokens = tokens[~mask].reshape(
                batch_size, num_tokens - mask[0].sum().item(),
                self.branch_channels)
            visible_positions = patch_positions.expand(batch_size, -1, -1)[~mask]
            visible_positions = visible_positions.reshape_as(visible_tokens)
            visible_sequence = torch.cat((class_tokens, visible_tokens), dim=1)
            visible_position_sequence = torch.cat((
                self.position_embedding[:, :1].expand(batch_size, -1, -1),
                visible_positions), dim=1)
            encoded_visible_sequence = self.encoder_norm(self.encoder(
                visible_sequence + visible_position_sequence))
            if classification_uses_mask:
                encoded_class = encoded_visible_sequence[:, 0]
            encoded_visible = encoded_visible_sequence[:, 1:]

            decoder_tokens = self.mask_token.expand(
                batch_size, num_tokens, -1).clone()
            decoder_tokens[~mask] = encoded_visible.reshape(
                -1, self.branch_channels)
            decoded = self.reconstruction_decoder(
                decoder_tokens + patch_positions)
            prediction = self.patch_prediction(self.decoder_norm(decoded))
            target = self.patchify(images, num_tokens).detach()
            reconstruction_loss = F.mse_loss(prediction[mask], target[mask])
        if encoded_class is None:
            raise RuntimeError("Classification token was not produced.")

        residual = self.zero_conv(encoded_class.unsqueeze(-1)).squeeze(-1)

        diagnostics = {
            "reconstruction_loss": reconstruction_loss,
            "cam": cam,
            "mask": mask,
            "classification_uses_mask": classification_uses_mask,
            "classification_token": encoded_class.detach(),
        }
        return residual, diagnostics
