from enum import Enum
import torch
import torch.nn as nn

class FeatMode(str, Enum):
    ALL   = "all"
    MAIN  = "main"
    EXTRA = "extra"
    GRID  = "grid"
    MAIN_EXTRA = "main+extra"
    MAIN_GRID  = "main+grid"
    EXTRA_GRID = "extra+grid"

def _batch_and_device_from_args(*args):
    for t in args:
        if isinstance(t, torch.Tensor):
            return t.size(0), t.device
    return None, None

class ChannelMaskWrapper(nn.Module):
    def __init__(self, base_model, mode: FeatMode, d_esf_norm=44, d_grid=27, d_main=704):
        super().__init__()
        self.m = base_model
        self.mode = FeatMode(mode)
        self.d_esf_norm = int(d_esf_norm)
        self.d_grid = int(d_grid)
        self.d_main = int(d_main)

        # aus Basismodell lesen (falls vorhanden)
        self.use_main     = getattr(base_model, "use_main",     True)
        self.use_esf_norm = getattr(base_model, "use_esf_norm", True)
        self.use_grid     = getattr(base_model, "use_grid",     True)

        # Konsistenzhinweis (optional streng machen)
        want_main  = self.mode in {FeatMode.ALL, FeatMode.MAIN, FeatMode.MAIN_EXTRA, FeatMode.MAIN_GRID}
        want_extra = self.mode in {FeatMode.ALL, FeatMode.EXTRA, FeatMode.MAIN_EXTRA, FeatMode.EXTRA_GRID}
        want_grid  = self.mode in {FeatMode.ALL, FeatMode.GRID,  FeatMode.MAIN_GRID,  FeatMode.EXTRA_GRID}
        if want_main  and not self.use_main:     print("[WARN] MODE will MAIN, base_model.use_main=False.")
        if want_extra and not self.use_esf_norm: print("[WARN] MODE will EXTRA, base_model.use_esf_norm=False.")
        if want_grid  and not self.use_grid:     print("[WARN] MODE will GRID, base_model.use_grid=False.")

    def _split_if_needed(self, *args):
        if len(args) == 3:
            return args
        if len(args) == 2:
            x704, xext = args
            if xext is None:
                return x704, None, None
            D = xext.size(1)
            if D >= (self.d_esf_norm + self.d_grid):
                return x704, xext[:, :self.d_esf_norm], xext[:, self.d_esf_norm:self.d_esf_norm+self.d_grid]
            elif D == self.d_esf_norm:
                return x704, xext, None
            elif D == self.d_grid:
                return x704, None, xext
            else:
                B, dev = xext.size(0), xext.device
                x44 = torch.zeros(B, self.d_esf_norm, device=dev)
                x27 = torch.zeros(B, self.d_grid,     device=dev)
                take44 = min(D, self.d_esf_norm); x44[:, :take44] = xext[:, :take44]
                if D > self.d_esf_norm:
                    take27 = min(D - self.d_esf_norm, self.d_grid)
                    x27[:, :take27] = xext[:, self.d_esf_norm:self.d_esf_norm+take27]
                return x704, x44, x27
        raise TypeError("Erwarte 2 oder 3 Inputs für forward.")

    @staticmethod
    def _apply_mode(x704, x44, x27, mode: FeatMode):
        if mode == FeatMode.ALL:         return x704, x44, x27
        if mode == FeatMode.MAIN:        return x704, None, None
        if mode == FeatMode.EXTRA:       return None, x44, None
        if mode == FeatMode.GRID:        return None, None, x27
        if mode == FeatMode.MAIN_EXTRA:  return x704, x44, None
        if mode == FeatMode.MAIN_GRID:   return x704, None, x27
        if mode == FeatMode.EXTRA_GRID:  return None, x44, x27
        raise ValueError(f"Unbekannter mode: {mode}")

    def forward(self, *args):
        B0, dev0 = _batch_and_device_from_args(*args)
        x704, x44, x27 = self._split_if_needed(*args)
        x704, x44, x27 = self._apply_mode(x704, x44, x27, self.mode)

        # Falls nach Maske alles None → Null-Fill für aktivierte Kanäle
        if (x704 is None) and (x44 is None) and (x27 is None):
            if B0 is None:
                raise ValueError("Alle Eingaben sind None. Batchgröße unbekannt.")
            if self.use_main:     x704 = torch.zeros(B0, self.d_main,     device=dev0)
            if self.use_esf_norm: x44  = torch.zeros(B0, self.d_esf_norm, device=dev0)
            if self.use_grid:     x27  = torch.zeros(B0, self.d_grid,     device=dev0)
            if (x704 is None) and (x44 is None) and (x27 is None):
                raise AssertionError("Keine aktiven Kanäle konfiguriert.")
        return self.m(x704, x44, x27)