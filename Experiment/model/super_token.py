from torch.nn import Module, ModuleList, Conv1d
from torch import Tensor
import torch

class SuperTokenEmbedding(Module):

    def __init__(self, embed_dim: int, kernel_sizes: list[int], each_kernel_size_output_dim: int):
        super().__init__()
        self.conv1d_list = ModuleList([
            Conv1d(
                in_channels=embed_dim,
                out_channels=each_kernel_size_output_dim,
                kernel_size=kernel_size,
                padding="same"
            ) for kernel_size in kernel_sizes
        ])

    def forward(self, pooled_encodings: list[Tensor]):
        pooled_encodings = [pooled_encoding.transpose(0, 1) for pooled_encoding in pooled_encodings]
        return [
            torch.cat([
                conv1d(pooled_encoding)
                for conv1d in self.conv1d_list
            ]).transpose(0, 1)
            for pooled_encoding in pooled_encodings
        ]
