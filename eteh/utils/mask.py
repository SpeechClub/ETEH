import torch
import numpy as np 
from distutils.version import LooseVersion

def make_pad_mask(lengths, xs=None, length_dim=-1, max_length=-1):
    """Function to make mask tensor containing indices of padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[0, 0, 0, 0 ,0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1]]

    :param list lengths: list of lengths (B)
    :param torch.Tensor xs: Make the shape to be like.
    :param int length_dim:
    :return: mask tensor containing indices of padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
        maxlen = max(maxlen, max_length)
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None
                    for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask

def subsequent_mask(size, device="cpu", dtype=torch.uint8):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    # ret = torch.ones(size, size, device=device, dtype=dtype)
    ret = torch.ones(size, size, dtype=dtype)
    return torch.tril(ret, out=ret).to(device=device)

def target_mask(ys_in_pad,ignore_id=-1):
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
        return ys_mask.unsqueeze(-2).bool() & m.bool()
    else:
        return ys_mask.unsqueeze(-2) & m

def ignore_fill(xs,x_len,ignore_id=-1):
    """Filled a Tensor with ignore_id where the index ai dim 1 is larger than the x_len

    :param torch.Tensor: input tensor [B,T,D]

    """
    assert xs.size(0)>x_len.size(0)
    ys = ignore_id*torch.ones_like(xs).to(xs.device)
    ys[:,:x_len] = xs[:,:x_len]
    return ys