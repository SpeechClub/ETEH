import json
import numpy as np
import itertools

def batchfy_by_seq(
        sorted_data, batch_size, max_length_in, max_length_out,
        min_batch_size=1, shortest_first=False, kaldi_data=False,
        ikey="input", okey="output"):
    """Make batch set from json dictionary

    :param Dict[str, Dict[str, Any]] sorted_data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse

    :param str ikey: key to access input (for ASR ikey="input", for TTS ikey="output".)
    :param str okey: key to access output (for ASR okey="output". for TTS okey="input".)
    :return: List[List[Tuple[str, dict]]] list of batches
    """
    if batch_size <= 0:
        raise ValueError("Invalid batch_size={batch_size}")

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        if not kaldi_data:
            _, info = sorted_data[start]
            ilen = int(info[ikey][0]['shape'][0])
            olen = int(info[okey][0]['shape'][0])
        else:
            ilen, _ = sorted_data[start].get_key_len_dim(ikey)
            olen, _ = sorted_data[start].get_key_len_dim(okey)
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i]
                                    for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # batch: List[List[Tuple[str, dict]]]
    return minibatches


def batchfy_by_bin(sorted_data, batch_bins, num_batches=0, min_batch_size=1, shortest_first=False, kaldi_data=False,
                   ikey="input", okey="output"):
    """Make variably sized batch set, which maximizes the number of bins up to `batch_bins`.

    :param Dict[str, Dict[str, Any]] sorted_data: dictionary loaded from data.json
    :param int batch_bins: Maximum frames of a batch
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param int test: Return only every `test` batches
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse

    :param str ikey: key to access input (for ASR ikey="input", for TTS ikey="output".)
    :param str okey: key to access output (for ASR okey="output". for TTS okey="input".)

    :return: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    if batch_bins <= 0:
        raise ValueError("invalid batch_bins={batch_bins}")
    length = len(sorted_data)
    if not kaldi_data:
        idim = int(sorted_data[0][1][ikey][0]['shape'][1]) if len(sorted_data[0][1][ikey][0]['shape']) > 1 else 1
        odim = int(sorted_data[0][1][okey][0]['shape'][1]) if len(sorted_data[0][1][okey][0]['shape']) > 1 else 1
    else:
        _, idim = sorted_data[0].get_key_len_dim(ikey)
        _, odim = sorted_data[0].get_key_len_dim(okey)
    #logging.info('# utts: ' + str(len(sorted_data)))
    minibatches = []
    start = 0
    n = 0
    while True:
        # Dynamic batch size depending on size of samples
        b = 0
        next_size = 0
        max_olen = 0
        while next_size < batch_bins and (start + b) < length:
            if not kaldi_data:
                ilen = int(sorted_data[start + b][1][ikey][0]['shape'][0]) * idim
                olen = int(sorted_data[start + b][1][okey][0]['shape'][0]) * odim
            else:
                ilen, _ = sorted_data[start + b].get_key_len_dim(ikey)
                olen, _ = sorted_data[start + b].get_key_len_dim(okey)
                ilen *= idim
                olen *= odim
            if olen > max_olen:
                max_olen = olen
            next_size = (max_olen + ilen) * (b + 1)
            if next_size <= batch_bins:
                b += 1
            elif next_size == 0:
                raise ValueError(
                    "Can't fit one sample in batch_bins ({batch_bins}): Please increase the value")
        end = min(length, start + max(min_batch_size, b))
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        if end == length:
            break
        start = end
        n += 1
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    #logging.info(str(len(minibatches)) + " batches containing from " +
    #             str(min(lengths)) + " to " + str(max(lengths)) + " samples " +
    #             "(avg " + str(int(np.mean(lengths))) + " samples).")
    return minibatches


def batchfy_by_frame(sorted_data, max_frames_in, max_frames_out, max_frames_inout,
                     num_batches=0, min_batch_size=1, shortest_first=False, kaldi_data=False,
                     ikey="input", okey="output"):
    """Make variably sized batch set, which maximizes the number of frames to max_batch_frame.

    :param Dict[str, Dict[str, Any]] sorteddata: dictionary loaded from data.json
    :param int max_frames_in: Maximum input frames of a batch
    :param int max_frames_out: Maximum output frames of a batch
    :param int max_frames_inout: Maximum input+output frames of a batch
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param int test: Return only every `test` batches
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse

    :param str ikey: key to access input (for ASR ikey="input", for TTS ikey="output".)
    :param str okey: key to access output (for ASR okey="output". for TTS okey="input".)

    :return: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    if max_frames_in <= 0 and max_frames_out <= 0 and max_frames_inout <= 0:
        raise ValueError(
            "At least, one of `--batch-frames-in`, `--batch-frames-out` or `--batch-frames-inout` should be > 0")
    length = len(sorted_data)
    minibatches = []
    start = 0
    end = 0
    while end != length:
        # Dynamic batch size depending on size of samples
        b = 0
        max_olen = 0
        max_ilen = 0
        while (start + b) < length:
            if not kaldi_data:
                ilen = int(sorted_data[start + b][1][ikey][0]['shape'][0])
                olen = int(sorted_data[start + b][1][okey][0]['shape'][0])
            else:
                ilen, _ = sorted_data[start + b].get_key_len_dim(ikey)
                olen, _ = sorted_data[start + b].get_key_len_dim(okey)
            if ilen > max_frames_in and max_frames_in != 0:
                raise ValueError(
                    "Can't fit one sample in --batch-frames-in ({max_frames_in}): Please increase the value")
            if olen > max_frames_out and max_frames_out != 0:
                raise ValueError(
                    "Can't fit one sample in --batch-frames-out ({max_frames_out}): Please increase the value")
            if ilen + olen > max_frames_inout and max_frames_inout != 0:
                raise ValueError(
                    "Can't fit one sample in --batch-frames-out ({max_frames_inout}): Please increase the value")
            max_olen = max(max_olen, olen)
            max_ilen = max(max_ilen, ilen)
            in_ok = max_ilen * (b + 1) <= max_frames_in or max_frames_in == 0
            out_ok = max_olen * (b + 1) <= max_frames_out or max_frames_out == 0
            inout_ok = (max_ilen + max_olen) * (b + 1) <= max_frames_inout or max_frames_inout == 0
            if in_ok and out_ok and inout_ok:
                # add more seq in the minibatch
                b += 1
            else:
                # no more seq in the minibatch
                break
        end = min(length, start + b)
        batch = sorted_data[start:end]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        start = end
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    lengths = [len(x) for x in minibatches]
    # logging.info(str(len(minibatches)) + " batches containing from " +
    #              str(min(lengths)) + " to " + str(max(lengths)) + " samples" +
    #              "(avg " + str(int(np.mean(lengths))) + " samples).")

    return minibatches


def batchfy_shuffle(data, batch_size, min_batch_size, num_batches, shortest_first):
    import random
    # logging.info('use shuffled batch.')
    sorted_data = random.sample(data.items(), len(data.items()))
    # logging.info('# utts: ' + str(len(sorted_data)))
    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        end = min(len(sorted_data), start + batch_size)
        # check each batch is more than minimum batchsize
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i] for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)
        if end == len(sorted_data):
            break
        start = end

    # for debugging
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
        # logging.info('# minibatches: ' + str(len(minibatches)))
    return minibatches


BATCH_COUNT_CHOICES = ["auto", "seq", "bin", "frame"]
BATCH_SORT_KEY_CHOICES = ["input", "output", "shuffle"]

MAX_ILEN = 1000000000
MAX_OLEN = 1000000000

def make_batchset(data, batch_size=0, max_length_in=float("inf"), max_length_out=float("inf"),
                  num_batches=0, min_batch_size=1, shortest_first=False, batch_sort_key="input", swap_io=False,
                  count="auto", batch_bins=0, batch_frames_in=0, batch_frames_out=0, batch_frames_inout=0,
                  ilen_max=MAX_ILEN, ilen_min=17, olen_max=MAX_OLEN, olen_min=1, down_sample=4, domain_id=None, clean_data=False):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as batchfy_by_{count}

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: maximum number of sequences in a minibatch.
    :param int batch_bins: maximum number of bins (frames x dim) in a minibatch.
    :param int batch_frames_in:  maximum number of input frames in a minibatch.
    :param int batch_frames_out: maximum number of output frames in a minibatch.
    :param int batch_frames_out: maximum number of input+output frames in a minibatch.
    :param str count: strategy to count maximum size of batch.
        For choices, see espnet.asr.batchfy.BATCH_COUNT_CHOICES

    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
        :return: List[List[Tuple[str, dict]]] list of batches
    :param str batch_sort_key: how to sort data before creating minibatches ["input", "output", "shuffle"]
    :param bool swap_io: if True, use "input" as output and "output" as input in `data` dict
    """

    # check args
    if count not in BATCH_COUNT_CHOICES:
        raise ValueError("arg 'count' ({count}) should be one of {BATCH_COUNT_CHOICES}")
    if batch_sort_key not in BATCH_SORT_KEY_CHOICES:
        raise ValueError("arg 'batch_sort_key' ({batch_sort_key}) should be one of {BATCH_SORT_KEY_CHOICES}")

    # for TTS
    # TODO(karita): remove this by creating converter from ASR to TTS json format
    if swap_io:
        ikey = "output"
        okey = "input"
        if batch_sort_key == "input":
            batch_sort_key = "output"
        elif batch_sort_key == "output":
            batch_sort_key = "input"
    else:
        ikey = "input"
        okey = "output"

    if count == "auto":
        if batch_size != 0:
            count = "seq"
        elif batch_bins != 0:
            count = "bin"
        elif batch_frames_in != 0 or batch_frames_out != 0 or batch_frames_inout != 0:
            count = "frame"
        else:
            raise ValueError("cannot detect `count` manually set one of {BATCH_COUNT_CHOICES}")
        # logging.info("count is auto detected as {count}")

    if count != "seq" and batch_sort_key == "shuffle":
        raise ValueError("batch_sort_key=shuffle is only available if batch_count=seq")

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        if domain_id is None or v['input'][0]['domain'] == domain_id:
            category2data.setdefault(v.get('category'), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for d in category2data.values():
        if batch_sort_key == 'shuffle':
            batches = batchfy_shuffle(d, batch_size, min_batch_size, num_batches, shortest_first)
            batches_list.append(batches)
            continue

        # sort it by input lengths (long to short)
        sorted_data = sorted(d.items(), key=lambda data: int(
            data[1][batch_sort_key][0]['shape'][0]), reverse=not shortest_first)
        if clean_data:
            sorted_data = CleanData(sorted_data, ilen_max=ilen_max, ilen_min=ilen_min, olen_max=olen_max, olen_min=olen_min, down_sample=down_sample)
        # logging.info('# utts: ' + str(len(sorted_data)))
        if count == "seq":
            batches = batchfy_by_seq(
                sorted_data,
                batch_size=batch_size,
                max_length_in=max_length_in,
                max_length_out=max_length_out,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey, okey=okey)
        if count == "bin":
            batches = batchfy_by_bin(
                sorted_data,
                batch_bins=batch_bins,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey, okey=okey)
        if count == "frame":
            batches = batchfy_by_frame(
                sorted_data,
                max_frames_in=batch_frames_in,
                max_frames_out=batch_frames_out,
                max_frames_inout=batch_frames_inout,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first,
                ikey=ikey, okey=okey)
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    # logging.info('# minibatches: ' + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches

def CleanData(data_list, ikey="input", okey="output", ilen_max=5000, ilen_min=17, olen_max=600, olen_min=1, down_sample=4, kaldi_data=False):
    new_data = []
    for i in range(len(data_list)):
        if not kaldi_data:
            _, info = data_list[i]
            ilen = int(info[ikey][0]['shape'][0])
            olen = int(info[okey][0]['shape'][0])
        else:
            ilen, _ = data_list[i].get_key_len_dim(ikey)
            olen, _ = data_list[i].get_key_len_dim(okey)            
        # Added by Gaochang Feng
        #3 Assuming sub-sampling factor is 4
        if ilen > ilen_max or (ilen - 3) // down_sample < 2 * olen + 1:
            continue
        if olen > olen_max:
            continue
        if ilen < ilen_min:
            continue
        if olen < olen_min:
            continue
        new_data.append(data_list[i])
    return new_data