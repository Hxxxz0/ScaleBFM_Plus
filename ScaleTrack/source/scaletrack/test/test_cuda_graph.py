import unittest

import torch

from my_rsl_rl.utils.cuda_graph import GraphedFunction


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class GraphedFunctionTests(unittest.TestCase):
    def test_replay_matches_eager_gradients_and_outputs(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).cuda()
        opt = torch.optim.Adam(net.parameters())

        def step(x, extra):
            opt.zero_grad(set_to_none=False)
            h = x
            for _ in range(4):  # a short unrolled rollout, as in imagination
                h = h + 0.1 * net(h).expand_as(h)
            loss = (h * extra["w"]).pow(2).mean()
            loss.backward()
            return {"loss": loss.detach(), "h": h.detach()}

        graphed = GraphedFunction(step, warmup=1)
        for i in range(4):
            x = torch.randn(16, 8, device="cuda")
            extra = {"w": torch.rand(16, 1, device="cuda")}
            out = graphed(x, extra)
            got = [p.grad.clone() for p in net.parameters()]
            ref = step(x, extra)
            for g, p in zip(got, net.parameters()):
                torch.testing.assert_close(g, p.grad)
            torch.testing.assert_close(out["loss"], ref["loss"])
            torch.testing.assert_close(out["h"], ref["h"])
            opt.step()  # parameters change in place between replays
        self.assertIsNotNone(graphed.graph)

    def test_checkpointed_compiled_attention_captures(self):
        """The SuperMimic actor step: bf16 fused attention, compiled, checkpointed outside the compiled region.

        Checkpointing inside torch.compile instead fails at capture (CUDAGeneratorImpl::current_seed)."""
        from torch.utils.checkpoint import checkpoint

        torch.manual_seed(0)
        att = torch.nn.MultiheadAttention(64, 4, batch_first=True).cuda()
        head = torch.nn.Linear(64, 8).cuda()
        params = [*att.parameters(), *head.parameters()]

        def actor_bf16(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, _ = att(x, x, x, need_weights=False)
                return head(y).mean(1).float()

        actor_fwd = torch.compile(actor_bf16, dynamic=False)

        def step(x):
            for p in params:
                p.grad = torch.zeros_like(p) if p.grad is None else p.grad.zero_()
            h, loss = x, 0.0
            for _ in range(4):  # several actor calls per graph, as in imagination
                a = checkpoint(actor_fwd, h, use_reentrant=False, preserve_rng_state=False)
                loss = loss + a.square().mean()
                h = h + a.mean(-1)[:, None, None]
            loss.backward()
            return loss.detach()

        graphed = GraphedFunction(step)
        x = torch.randn(32, 16, 64, device="cuda")
        for _ in range(5):
            out = graphed(x)
        self.assertIsNotNone(graphed.graph)
        got = [p.grad.clone() for p in params]
        ref = step(x)
        torch.testing.assert_close(out, ref)
        for g, p in zip(got, params):
            torch.testing.assert_close(g, p.grad)


if __name__ == "__main__":
    unittest.main()
