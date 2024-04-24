from enum import Enum
import numpy as np
import torch
from spikenet.layers.spiking_base import SpikingNeuron
from spikenet.tools.configs import EPSILON


class SpikingDenseLayer(SpikingNeuron):
    class TimeReduction(Enum):
        NoTimeReduction = None
        SpikeRate = "SpikeRate"
        SpikeTime = "SpikeTime"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.w = torch.nn.Parameter(
            torch.empty((self.input_dim, self.output_dim)), requires_grad=True
        )
        self.beta = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(self.output_dim), requires_grad=True)
        self.__time_reduction = self.TimeReduction(kwargs.get("time_reduction", None))

    def __apply_time_reduction(self, x: torch.Tensor) -> torch.Tensor:
        match self.__time_reduction:
            case self.TimeReduction.NoTimeReduction:
                return x
            case self.TimeReduction.SpikeRate:
                return self.__spike_rate_reduction(x)
            case self.TimeReduction.SpikeTime:
                return self.__spike_time_reduction(x)
        raise ValueError("Invalid time_reduction")

    def __spike_rate_reduction(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.max(x, 1)[0]
        return res

    def __spike_time_reduction(self, x: torch.Tensor) -> torch.Tensor:
        max_data = torch.max(x, 1)[1]
        max_min_data = max_data.min(1)[1]
        return torch.nn.functional.one_hot(
            max_min_data, num_classes=self.output_dim
        ).to(torch.float32)

    def initialize_parameters(self) -> None:
        torch.nn.init.normal_(
            self.w,
            mean=self.w_init_mean,
            std=self.w_init_std * np.sqrt(1.0 / self.input_dim),
        )
        torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)
        torch.nn.init.normal_(self.b, mean=1.0, std=0.01)
        assert not torch.isnan(self.w).any(), "NaN in w"
        assert not torch.isnan(self.beta).any(), "NaN in beta"
        assert not torch.isnan(self.b).any(), "NaN in b"

    def clamp(self) -> None:
        self.beta.data.clamp_(0.0, 1.0)
        self.b.data.clamp_(min=0.0)
        assert not torch.isnan(self.beta).any(), "NaN in beta"
        assert not torch.isnan(self.b).any(), "NaN in b"

    def __str__(self) -> str:
        # TODO: add time_reduction
        return f"D({self.output_dim})"

    def forward(self, x: torch.Tensor, save_history: bool = True) -> torch.Tensor:
        res = self._spiking_forward(x, save_history)
        return self.__apply_time_reduction(res)

    # TODO: find better name
    def _spiking_forward(self, x: torch.Tensor, save_history: bool) -> torch.Tensor:
        batch_size = x.shape[0]
        nb_steps = x.shape[1]
        if save_history:
            self._reset_history() # TODO: maybe remove this line
            self._set_time_scale(nb_steps)

        assert not torch.isnan(x).any(), "NaN in x"
        assert not torch.isnan(self.w).any(), "NaN in w"

        h = torch.einsum("abc,cd->abd", x, self.w)
        assert not torch.isnan(h).any(), "NaN in h"

        # membrane potential
        mem = torch.zeros(
            (batch_size, self.output_dim), dtype=x.dtype, device=x.device
        )
        spk = torch.zeros(
            (batch_size, self.output_dim), dtype=x.dtype, device=x.device
        )

        # output spikes recording
        spk_rec = torch.zeros(
            (batch_size, nb_steps, self.output_dim), dtype=x.dtype, device=x.device
        )
        norm = (self.w**2).sum(0)
        assert not torch.isnan(norm).any()

        self._reset_history()
        for t in range(nb_steps):
            # reset term
            rst = spk * self.b * norm
            assert not torch.isnan(rst).any(), "NaN in rst"
            input_ = h[:, t, :]
            assert not torch.isnan(input_).any(), "NaN in input_"

            mem = (mem - rst) * self.beta + input_ * (1.0 - self.beta)
            assert not torch.isnan(mem).any(), "NaN in mem"
            mthr = torch.einsum("ab,b->ab", mem, 1.0 / (norm + EPSILON)) - self.b
            assert not torch.isnan(mthr).any(), "NaN in mthr"
            spk = self.spike_fn(mthr)
            assert not torch.isnan(spk).any(), "NaN in spk"

            # assert 0 <= mem.min() <= mem.max() <= 1, f"mem out of range {mem.min()} << {mem.max()}"

            spk_rec[:, t, :] = spk

            if save_history:
                self._history("mem", t, mem)
                self._history("mthr", t, mthr)
                self._history("spk", t, spk)

        return spk_rec

    def snapshot_cycle(self) -> None:
        self._append_snapshot("w", self.w.detach().cpu().numpy())
        self._append_snapshot("beta", self.beta.detach().cpu().numpy())
        self._append_snapshot("b", self.b.detach().cpu().numpy())
