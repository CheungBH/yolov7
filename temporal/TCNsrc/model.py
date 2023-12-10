import torch.nn.functional as F
from torch import nn
from .tcn import TemporalConvNet

TCN_single = False

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, dilation):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, dilation=dilation)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        #y1 = self.linear(y1)
        o = self.linear(y1[:, :, -1])
        if TCN_single:
            return (F.sigmoid(o))*o
        return F.log_softmax(o, dim=1)
        #print((F.sigmoid(o))*o)

        #return (F.sigmoid(o))*o