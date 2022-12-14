import torch

class VGG2L(torch.nn.Module):
    """VGG2L module for transformer encoder.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs

    """

    def __init__(self, idim, odim, domain_dim=0):
        """Construct a VGG2L object."""
        super().__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        self.output = torch.nn.Linear(128 * ((idim // 2) // 2) + domain_dim, odim)

    def forward(self, x, x_mask, x_tag=None):
        """VGG2L forward for x.

        Args:
            x (torch.Tensor): input torch (B, T, idim)
            x_mask (torch.Tensor): (B, 1, T)

        Returns:
            x (torch.Tensor): input torch (B, sub(T), attention_dim)
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x = x.unsqueeze(1)
        x = self.vgg2l(x)

        b, c, t, f = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        if x_tag is not None:
            x= torch.cat((x, x_tag.unsqueeze(1).repeat(1,x.size(1),1)), dim=-1)

        x = self.output(x)

        if x_mask is not None:
            x_mask = self.create_new_mask(x_mask, x)

        return x, x_mask

    def create_new_mask(self, x_mask, x):
        """Create a subsampled version of x_mask.

        Args:
            x_mask (torch.Tensor): (B, 1, T)
            x (torch.Tensor): (B, sub(T), attention_dim)

        Returns:
            x_mask (torch.Tensor): (B, 1, sub(T))

        """
        x_t1 = x_mask.size(2) - (x_mask.size(2) % 3)
        x_mask = x_mask[:, :, :x_t1][:, :, ::3]

        x_t2 = x_mask.size(2) - (x_mask.size(2) % 2)
        x_mask = x_mask[:, :, :x_t2][:, :, ::2]

        return x_mask