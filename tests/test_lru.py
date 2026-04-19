import unittest

import torch
from torch import nn

import torch_structured  # noqa: F401 — triggers extension load
from torch_structured import LRU
from torch_structured.recurrent import lru as _lru_mod


class LRUTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_lru_matches_nn_gru_interface(self) -> None:
        H, T, B = 16, 8, 2
        grid = [
            (1, False, False),
            (1, False, True),
            (2, True, True),
            (2, False, False),
        ]
        for num_layers, bidirectional, batch_first in grid:
            with self.subTest(num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first):
                D = 2 if bidirectional else 1
                x = (torch.randn(B, T, H) if batch_first
                     else torch.randn(T, B, H))
                h0 = torch.zeros(num_layers * D, B, H)
                gru = nn.GRU(H, H, num_layers=num_layers,
                             bidirectional=bidirectional,
                             batch_first=batch_first)
                lru = LRU(H, H, num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first, kind='dense')
                g_out, g_hn = gru(x, h0)
                l_out, l_hn = lru(x, h0)
                self.assertEqual(tuple(l_out.shape), tuple(g_out.shape))
                self.assertEqual(tuple(l_hn.shape), tuple(g_hn.shape))

    def test_lru_forward_backward(self) -> None:
        m = LRU(16, 16, num_layers=1, batch_first=True)
        x = torch.randn(2, 8, 16)
        out, _ = m(x)
        out.sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_lru_structured_kind(self) -> None:
        H, T, B = 16, 8, 2
        x = torch.randn(B, T, H)
        ref = LRU(H, H, num_layers=1, batch_first=True, kind='dense')
        ref_out, _ = ref(x)
        for kind in ('butterfly', 'monarch'):
            with self.subTest(kind=kind):
                m = LRU(H, H, num_layers=1, batch_first=True, kind=kind)
                out, _ = m(x)
                self.assertEqual(tuple(out.shape), tuple(ref_out.shape))

    def test_lru_scan_vs_naive(self) -> None:
        if not _lru_mod._HAS_SCAN:
            self.skipTest("torch.associative_scan unavailable")
        m = LRU(16, 16, num_layers=1, batch_first=True)
        x = torch.randn(2, 8, 16)
        out_scan, _ = m(x)
        try:
            _lru_mod._FORCE_NAIVE = True
            out_naive, _ = m(x)
        finally:
            _lru_mod._FORCE_NAIVE = False
        self.assertTrue(torch.allclose(out_scan, out_naive,
                                       atol=1e-4, rtol=1e-4),
                        (out_scan - out_naive).abs().max().item())

    def test_lru_bidirectional_concat(self) -> None:
        H = 16
        m = LRU(H, H, num_layers=1, bidirectional=True,
                batch_first=True, kind='dense')
        x = torch.randn(2, 8, H)
        bi_out, _ = m(x)
        fwd, bwd = bi_out[..., :H], bi_out[..., H:]
        self.assertFalse(torch.allclose(fwd, bwd, atol=1e-6),
                         "forward and reverse halves must differ")


if __name__ == '__main__':
    unittest.main()
