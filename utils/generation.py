import torch
from torch.nn import functional as F


def generate_greedy(sequence, config, forward, inv_vocab, n = 10):
    """This function takes in a sequence, and for ``n`` steps generates tokens, and replaces sequence
    with the highest probability token."""
    assert n <= config.input_len, f"Cannot generate ({n}) more tokens than in the input sequence ({config.input_len})"

    filled_loc = set()
    for i in range(n):
        out = forward(sequence)
        values, indices = out.topk(1)
        index_sorted = values.argsort(1, descending=True).squeeze().tolist()
        curr_index=0
        for index in range(len(index_sorted)):
            if index_sorted[index] not in filled_loc:
                filled_loc.add(index_sorted[index])
                curr_index=index
                break
        sequence = sequence[:i] + inv_vocab[indices[0, curr_index, 0].item()] + sequence[i+1:]
    return sequence


def generate_topk(sequence,config,forward,inv_vocab,n=10,k=2):
    """This function takes in a sequence, and for ``n`` steps generates tokens, and replaces sequence
    with the k highest probability tokens."""
    assert n <= config.input_len, f"Cannot generate ({n}) more tokens than in the input sequence ({config.input_len})"
    filled_loc = set()
    for i in range(n):
        # for this step in generation
        out = forward(sequence)
        values, indices = out.topk(1)
        print(indices)
        index_sorted = values.argsort(1, descending=True).squeeze().tolist()
        k_value=k
        curr_index=0
        remove_index=[]
        while(k_value):
            if len(index_sorted)<k:
                continue
            while(index_sorted[curr_index] in filled_loc):
                del index_sorted[curr_index]
                curr_index+=1
            remove_index.append(curr_index)
            k_value-=1
        for index in range(0,k):
            sequence = sequence[:i] + inv_vocab[indices[0, remove_index[index], 0].item()] + sequence[i+1:]
    return sequence


def generate_topP(sequence,config,forward,inv_vocab,n=10,p=0.8,min_tokens=1):
    """This function takes in a sequence, and for ``n`` steps generates tokens, and replaces sequence
    with the top tokens with cumulative probability>=p"""
    assert n <= config.input_len, f"Cannot generate ({n}) more tokens than in the input sequence ({config.input_len})"

    for i in range(n):
        out = forward(sequence)
        values, indices = torch.sort(out[0],descending=True)
        values = torch.Tensor([x[0] for x in values])
        indices = torch.LongTensor([x[0] for x in indices])
        cumulative_probs = torch.cumsum(F.softmax(values, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs> p
        print(sorted_indices_to_remove)
        if min_tokens > 1:
            sorted_indices_to_remove[..., :min_tokens] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        result_sequence=""
        for i in range(len(sorted_indices_to_remove)):
            if sorted_indices_to_remove[i]:
                result_sequence+=inv_vocab[indices[i].item()]
            else:
                result_sequence+=sequence[i]
        sequence=result_sequence
    return sequence


def generate_sample(sequence,config,forward,inv_vocab,n=10):
    assert n <= config.input_len, f"Cannot generate ({n}) more tokens than in the input sequence ({config.input_len})"
    for i in range(n):
        out = forward(sequence)
        probs = F.softmax(out, dim=-1)
        ix = torch.multinomial(probs[0], num_samples=1)
        result_sequence=""
        for i in range(len(ix)):
            result_sequence+=inv_vocab[ix[i][0].item()]
        sequence=result_sequence
    return sequence
