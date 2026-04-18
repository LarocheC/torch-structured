import math
import numbers

import torch
from torch import nn

from .butterfly import Butterfly
from .multiply_base4 import butterfly_multiply_base4_torch
from .multiply_base4 import twiddle_base2_to_base4


class ButterflyBase4(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.
    """

    def __init__(self, *args, **kwargs):
        init = kwargs.get('init', None)
        if (isinstance(init, tuple) and len(init) == 2 and isinstance(init[0], torch.Tensor)
            and isinstance(init[1], torch.Tensor)):
            twiddle4, twiddle2 = init[0].clone(), init[1].clone()
            kwargs['init'] = 'empty'
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        del self.twiddle
        self.twiddle4 = nn.Parameter(twiddle4)
        self.twiddle2 = nn.Parameter(twiddle2)
        self.twiddle4._is_structured = True
        self.twiddle2._is_structured = True

    def forward(self, input):
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        output = butterfly_multiply_base4_torch(self.twiddle4, self.twiddle2, output,
                                                self.increasing_stride, output_size)
        return self.post_process(input, output)

    def __imul__(self, scale):
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        scale_per_entry = scale ** (1.0 / self.nblocks / self.log_n)
        self.twiddle4 *= scale_per_entry ** 2
        self.twiddle2 *= scale_per_entry
        return self
