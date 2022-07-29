import numpy


def pad_list(xs, pad_value):
    """Function to pad values

    :param list xs: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :param float pad_value: value for padding
    :return: padded tensor (B, Lmax, D)
    :rtype: torch.Tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad

def calcurate_cer(xs_pre,label,ignore_id = -1):
    xs_label = numpy.argmax(xs_pre,axis=-1)
    xs_label = xs_label.reshape(-1)
    label = label.reshape(-1)
    x_corr = xs_label == label 
    x_corr = x_corr[label!=ignore_id].astype(numpy.float)
    return float(numpy.mean(x_corr))


def calculate_cer_ctc(xs_pre, ys_pad, idx_blank=0, idx_space=-1, idx_append=-1, xs_len=None):
    """Calculate sentence-level CER score for CTC.

    :param torch.Tensor ys_hat: prediction (batch, seqlen)
    :param torch.Tensor ys_pad: reference (batch, seqlen)
    :return: average sentence-level CER score
    :rtype float
    """
    from itertools import groupby
    import editdistance
    if xs_len is None:
        xs_len = [xs_pre.shape[1]] * xs_pre.shape[0]
    xs_label = numpy.argmax(xs_pre,axis=-1)
    cers, char_ref_lens = [], []
    for i, y in enumerate(xs_label):
        y_hat = [x[0] for x in groupby(y)]
        y_true = ys_pad[i]
        seq_hat, seq_true = [], []
        for idx in y_hat[:xs_len[i]]:
            idx = int(idx)
            if idx != idx_append and idx != idx_blank and idx != idx_space:
                seq_hat.append(int(idx))

        for idx in y_true:
            idx = int(idx)
            if idx != idx_append and idx != idx_blank and idx != idx_space:
                seq_true.append(int(idx))

        if len(seq_true) > 0:
            cers.append(editdistance.eval(seq_hat, seq_true))
            char_ref_lens.append(len(seq_true))
    cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else 0.0
    return cer_ctc


def to_device(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    import torch 
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)