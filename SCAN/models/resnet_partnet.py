"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn as nn
import numpy as np
import torch
import torchvision.models as models

# class SoftPositionEmbed(nn.Module):
#     def __init__(self, hidden_size, resolution):
#         """Builds the soft position embedding layer.
#         Args:
#         hidden_size: Size of input feature dimension.
#         resolution: Tuple of integers specifying width and height of grid.
#         """
#         super().__init__()
#         self.embedding = nn.Linear(4, hidden_size, bias=True)
#         self.grid = build_grid(resolution)

#     def forward(self, inputs):
#         grid = self.embedding(self.grid)
#         grid = grid.permute(0, 3, 1, 2)
#         return inputs + grid

# def build_grid(resolution):
#     ranges = [np.linspace(0., 1., num=res) for res in resolution]
#     grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
#     grid = np.stack(grid, axis=-1)
#     grid = np.reshape(grid, [resolution[0], resolution[1], -1])
#     grid = np.expand_dims(grid, axis=0)
#     grid = grid.astype(np.float32)
#     return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to('cuda')

def resnet18():
    backbone = models.resnet18()
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    backbone.fc = nn.Identity()
    return {'backbone': backbone, 'dim': 512}
