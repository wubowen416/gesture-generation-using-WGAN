# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.jit as jit
import torch.nn as nn
from torch.nn import Parameter
from typing import List, Tuple
from torch import Tensor
import math

# ----------------------------------------------------------------------------------------------------------------------
class JitGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(JitGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tensor
        x = x.view(-1, x.size(1))
        x_results = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        h_results = torch.mm(hidden, self.weight_hh.t()) + self.bias_hh

        i_r, i_z, i_n = x_results.chunk(3, 1)
        h_r, h_z, h_n = h_results.chunk(3, 1)

        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)

        return n - torch.mul(n, z) + torch.mul(z, hidden)

# ----------------------------------------------------------------------------------------------------------------------
class JitGRULayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(JitGRULayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, x, hidden):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = x.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])

        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden)
            outputs += [hidden]

        return torch.stack(outputs), hidden

# ----------------------------------------------------------------------------------------------------------------------
class JitGRU(jit.ScriptModule):
    __constants__ = ['hidden_size', 'num_layers', 'batch_first']

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, bias=True):
        super(JitGRU, self).__init__()
        # The following are not implemented.
        assert bias

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if num_layers == 1:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)])
        else:
            self.layers = nn.ModuleList([JitGRULayer(JitGRUCell, input_size, hidden_size)] + [JitGRULayer(JitGRUCell, hidden_size, hidden_size)
                                                                                              for _ in range(num_layers - 1)])

    @jit.script_method
    def forward(self, x, h=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        output_states = jit.annotate(List[Tensor], [])

        # Handle batch_first cases
        if self.batch_first:
            x = x.permute(1, 0, 2)

        if h is None:
            h = torch.zeros(self.num_layers, x.shape[1], self.hidden_size, dtype=x.dtype, device=x.device)

        output = x
        i = 0

        for rnn_layer in self.layers:
            output, hidden = rnn_layer(output, h[i])
            output_states += [hidden]
            i += 1

        # Don't forget to handle batch_first cases for the output too!
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, torch.stack(output_states)

# ----------------------------------------------------------------------------------------------------------------------
def test_script_gru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    h = torch.randn(batch, hidden_size)

    # The JIT GRU instance
    gru_jit = JitGRULayer(JitGRUCell, input_size, hidden_size)
    # Control: PyTorch's native GRU
    gru_pytorch = nn.GRU(input_size, hidden_size, 1)

    # The name of each JitGRU parameter that we've defined in JitGRUCell
    param_names = ["weight_hh", "weight_ih", "bias_ih", "bias_hh"]

    for param_name in param_names:
        # Build the name of the parameters in this layer
        jit_name = F"cell.{param_name}"
        pytorch_name = F"{param_name}_l0"

        jit_param = gru_jit.state_dict()[jit_name]
        gru_param = gru_pytorch.state_dict()[pytorch_name]

        # Make sure the shapes are the same
        assert gru_param.shape == jit_param.shape

        # Copy the weight values
        with torch.no_grad():
            gru_param.copy_(jit_param)

    # Run both on the same input
    out, out_state = gru_jit(inp, h)
    gru_out, gru_out_hidden = gru_pytorch(inp, h.unsqueeze(0))

    # Make sure the values are reasonably close
    assert (out - gru_out).abs().max() < 1e-5
    assert (out_state - gru_out_hidden).abs().max() < 1e-5

# ----------------------------------------------------------------------------------------------------------------------
def test_script_stacked_gru(seq_len, batch, input_size, hidden_size,
                            num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    gru_hidden = torch.stack([torch.randn(batch, hidden_size) for _ in range(num_layers)])

    # The JIT GRU instance
    gru_jit = JitGRU(input_size, hidden_size, num_layers)
    # Control: PyTorch's native GRU
    gru_pytorch = nn.GRU(input_size, hidden_size, num_layers)

    # The name of each JitGRU parameter that we've defined in JitGRUCell
    param_names = ["weight_hh", "weight_ih", "bias_ih", "bias_hh"]

    for layer in range(num_layers):
        for param_name in param_names:
            # Build the name of the parameters in this layer
            jit_name = F"layers.{layer}.cell.{param_name}"
            pytorch_name = F"{param_name}_l{layer}"

            jit_param = gru_jit.state_dict()[jit_name]
            gru_param = gru_pytorch.state_dict()[pytorch_name]

            # Make sure the shapes are the same
            assert gru_param.shape == jit_param.shape

            # Copy the weight values
            with torch.no_grad():
                gru_param.copy_(jit_param)

    # Run both on the same input
    out_jit, out_jit_hidden = gru_jit(inp, gru_hidden)
    out_pytorch, out_pytorch_hidden = gru_pytorch(inp, gru_hidden)

    # Make sure the values are reasonably close
    assert (out_jit - out_pytorch).abs().max() < 1e-5
    assert (out_jit_hidden - out_pytorch_hidden).abs().max() < 1e-5

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Testing layers...')
    test_script_gru_layer(5, 2, 3, 7)
    test_script_gru_layer(1, 1, 1, 1)
    print('Testing stacked GRUs...')
    test_script_stacked_gru(5, 2, 3, 7, 10)
    test_script_stacked_gru(1, 1, 1, 1, 1)
    print('All unit tests passed!')