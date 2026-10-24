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


if __name__ == "__main__":
    unittest.main()
