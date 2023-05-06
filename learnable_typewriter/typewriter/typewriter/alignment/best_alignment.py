import os
import torch
from torch.utils.cpp_extension import load
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot

module_path = os.path.dirname(__file__)
imputer = load("imputer_fn", sources=[os.path.join(module_path, f) for f in ["best_alignment.cpp", "best_alignment.cu", "imputer_loss.cu"]])

def get_alignment_path(log_alpha, path):
    if log_alpha.shape[0] == 1:
        current_state = 0
    else:
        current_state = log_alpha[-2:, -1].argmax() + (log_alpha.shape[0] - 2)

    path_decode = [current_state]
    for t in range(path.shape[1] - 1, 0, -1):
        prev_state = path[current_state, t]
        path_decode.append(prev_state)
        current_state = prev_state

    return path_decode[::-1]

def best_alignment(log_prob, targets, input_lengths, target_lengths, blank, zero_infinity=False):
    """Get best alignment (maximum probability sequence of ctc states)
       conditioned on log probabilities and target sequences

    Input:
        log_prob (T, N, C): C = number of characters in alphabet including blank
                            T = input length
                            N = batch size
                            log probability of the outputs (e.g. torch.log_softmax of logits)
        targets (N, S): S = maximum number of characters in target sequences
        input_lengths (N): lengths of log_prob
        target_lengths (N): lengths of targets
        blank (int): index of blank tokens (default 0)
        zero_infinity (bool): if true imputer loss will zero out infinities.
                            infinities mostly occur when it is impossible to generate
                            target sequences using input sequences
                            (e.g. input sequences are shorter than target sequences)

    Output:
        best_aligns (List[List[int]]): sequence of ctc states that have maximum probabilties
                                       given log probabilties, and compatible with target sequences"""
    nll, log_alpha, alignment = imputer.best_alignment(log_prob, targets, input_lengths, target_lengths, blank, zero_infinity)
    log_alpha = log_alpha.transpose(1, 2).detach().cpu().numpy()
    alignment = alignment.transpose(1, 2).detach().cpu().numpy()

    best_aligns = []
    for log_a, align, input_len, target_len in zip(log_alpha, alignment, input_lengths, target_lengths):
        state_len = target_len * 2 + 1
        log_a = log_a[:state_len, :input_len]
        align = align[:state_len, :input_len]

        best_aligns.append(get_alignment_path(log_a, align))

    return unify(best_aligns, log_prob, targets, blank)

def unify(best_aligns, log_prob, targets, blank):
    x = [torch.LongTensor(aggregate_local_max((get_symbol(p, targets[i], blank) for p in ps), log_prob.transpose(0, 1)[i], blank)) for i, ps in enumerate(best_aligns)]
    x = pad_sequence(x, padding_value=blank)
    x = one_hot(x, num_classes=log_prob.size(-1))
    return x.to(device=log_prob.device)

def aggregate_local_max(indexes, log_prob, blank):
    output, prev, prev_prob = [], -1, float('-inf')
    for j, i in enumerate(indexes):
        # check if the current element is the same as the previous one
        if i != blank and i == prev:
            # if yes, check if the current probability is higher than the previous one
            if log_prob[j, i] > prev_prob:
                # if yes, replace the previous element with a blank symbol
                output[-1] = blank
                prev_prob = log_prob[j, i]
            else:
                # otherwise, replace the current element with a blank symbol
                i = blank
        else:
            # if the current element is different from the previous one, set prev_prob to 0
            prev_prob = log_prob[j, i]

        output.append(i)
        prev = i
    return output


def get_symbol(state, targets_list, blank):
    """Convert sequence of ctc states into sequence of target tokens

    Input:
        state (List[int]): list of ctc states (e.g. from torch_imputer.best_alignment)
        targets_list (List[int]): token indices of targets
                                  (e.g. targets that you will pass to ctc_loss or imputer_loss)
    """
    return (blank if state % 2 == 0 else targets_list[state // 2].item())