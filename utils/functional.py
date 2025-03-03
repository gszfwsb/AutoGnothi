from typing import Callable, Iterable, List, Optional

import torch


def batched(
    inp: Callable[[], Iterable[torch.Tensor]],
    decorator: Callable[[torch.Tensor], torch.Tensor],
    operation: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> Iterable[torch.Tensor]:
    """Apply operation(s) to a number of tensors in batches. The data are
    pipelined through torch operators as may be inferred from the dimensions:

    inp: yield Tensor[<...dims_inp>];
    decorator: (Tensor[<...dims_inp>]) -> Tensor[x, <...dims_decorated>];
    operation: (Tensor[n_items <= sum of x, <...dims_decorated>]) -> Tensor[n_items, <...dims_out>];
    ret [*]: Tensor[x, <...dims_out>] for each inp."""

    in_sizes: List[int] = []  # [p?]
    buffer_in: List[torch.Tensor] = []  # <p?, ...dims_decorated>
    buffer_out: List[torch.Tensor] = []  # <q?, ...dims_out>

    def _flush_in(limit: int) -> bool:
        nonlocal buffer_in
        if not buffer_in:
            return False
        tot_size = sum(t.shape[0] for t in buffer_in)
        if not limit:
            limit = tot_size
        if tot_size < limit:
            return False
        sizes = [t.shape[0] for t in buffer_in]

        in_sizes.extend(list(sizes))
        stacked = torch.cat(buffer_in, dim=0)
        if tot_size > limit:
            sizes[-1] -= tot_size - limit
            stacked, rest = stacked.split([limit, tot_size - limit], dim=0)
            buffer_in = [rest]
        else:
            buffer_in = []

        out = operation(stacked)
        outs = out.split(sizes, dim=0)
        buffer_out.extend(outs)
        return True

    def _flush_ins(limit: int) -> None:
        while True:
            if not _flush_in(limit):
                break

    def _flush_out() -> Optional[torch.Tensor]:
        if not in_sizes:
            return None
        front_sz = in_sizes[0]
        tot_sz = 0

        chunks: List[torch.Tensor] = []
        for chunk in buffer_out:
            tot_sz += chunk.shape[0]
            chunks.append(chunk)
            if tot_sz >= front_sz:
                break
        if tot_sz < front_sz:
            return None

        for _ in range(len(chunks)):
            in_sizes.pop(0)
            buffer_out.pop(0)
        return torch.cat(chunks, dim=0)

    def _flush_outs() -> List[torch.Tensor]:
        outs = []
        while True:
            out = _flush_out()
            if out is None:
                break
            outs.append(out)
        return outs

    for _inp in inp():
        _dec = decorator(_inp)
        buffer_in.append(_dec)
        _flush_ins(batch_size)
        for out in _flush_outs():
            yield out
    # flush remaining
    _flush_ins(0)
    for out in _flush_outs():
        yield out
    return
