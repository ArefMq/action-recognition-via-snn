import torch

from scnn.Spike.readout import ReadoutLayer


class ReadoutStream(ReadoutLayer):
    def __init__(self, *args, **kwargs):
        super(ReadoutStream, self).__init__(*args, **kwargs)
        self.histogram_memory_size = kwargs.get('histogram_memory_size', 50)
        self.mem = None
        self.history_counter = 0

    def __str__(self):
        return 'stream(' + str(self.output_shape) + ')'

    def forward_function(self, x):
        batch_size = x.shape[0]
        if self.mem is None:
            self.reset_mem(batch_size, x.device, x.dtype)
        h = torch.einsum("ac,cd->ad", x, self.w)
        norm = (self.w ** 2).sum(0)

        # membrane potential update
        self.mem = self.mem * self.beta + (1 - self.beta) * h[:, :]
        self.mem_rec_hist[:, self.history_counter, :] = self.mem.detach().cpu()
        self.history_counter += 1
        if self.history_counter >= self.histogram_memory_size:
            self.history_counter = 0

        output = self.mem / (norm + 1e-8) - self.b
        return output

    def reset_mem(self, batch_size, x_device, x_dtype):
        self.mem = torch.zeros((batch_size, self.output_shape), dtype=x_dtype, device=x_device)
        self.mem_rec_hist = torch.zeros((batch_size, self.histogram_memory_size, self.output_shape), dtype=x_dtype)
        self.history_counter = 0

