"""Replay a fixed-shape GPU computation (forward and backward included) as a single CUDA graph.

Compiled imagination steps are still launch bound: thousands of short kernels per update keep the GPU waiting on
the Python host. Capturing the whole computation once and replaying it removes the per-kernel host cost.

The function must be free of host synchronisation and keep its tensor shapes. Inputs are copied into static
buffers before every replay; the returned outputs are static as well and are overwritten by the next replay.
Gradients accumulate into the ``.grad`` tensors that existed at capture time: the function zeroes them in place
(``zero_grad(set_to_none=False)``) and callers must not release them.
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class GraphedFunction:
    def __init__(self, fn, warmup: int = 3, enabled: bool = True):
        self.fn = fn
        self.warmup = warmup
        self.enabled = enabled
        self.calls = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.stream: torch.cuda.Stream | None = None

    def __call__(self, *args):
        if not self.enabled:
            return self.fn(*args)
        if self.graph is not None:
            torch._foreach_copy_(self.static_inputs, tree_flatten(args)[0])
            self.graph.replay()
            return self.static_outputs
        # Every execution before the replays (eager warm-up: compilation, lazy initialisation, gradient tensors;
        # capture) runs on one private stream. Autograd synchronises a parameter's gradient accumulation with the
        # stream of its first forward use, which therefore must be the capture stream rather than the default one.
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if self.calls < self.warmup:
                self.calls += 1
                out = self.fn(*args)
            else:
                out = self._capture(*tree_flatten(args))
        torch.cuda.current_stream().wait_stream(self.stream)
        return out

    def _capture(self, flat, spec):
        self.static_inputs = [t.clone() for t in flat]
        static_args = tree_unflatten(self.static_inputs, spec)
        graph = torch.cuda.CUDAGraph()
        # Other host threads (simulator) keep using CUDA while this one captures.
        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode="thread_local"):
            self.static_outputs = self.fn(*static_args)
        self.graph = graph
        graph.replay()  # capture records without executing
        return self.static_outputs
