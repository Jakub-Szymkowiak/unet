import torch.nn as nn

from unet_modules import Input, Contracting, Expansive, Output

class U_Net(nn.Module):
  def __init__(self, in_size, N):
    super().__init__()
    self.inp = Input(in_size)

    self.contr1 = Contracting(64,  128 )
    self.contr2 = Contracting(128, 256 )
    self.contr3 = Contracting(256, 512 )
    self.contr4 = Contracting(512, 1024)

    self.exp1 = Expansive(1024, 512)
    self.exp2 = Expansive(512 , 256)
    self.exp3 = Expansive(256 , 128)
    self.exp4 = Expansive(128 , 64 )

    self.out = Output(N)

  def forward(self, x):
    x_in = self.inp(x)
    
    x_contr1 = self.contr1(x_in)
    x_contr2 = self.contr2(x_contr1)
    x_contr3 = self.contr3(x_contr2)
    x_contr4 = self.contr4(x_contr3)

    x = self.exp1(x_contr4, x_contr3)
    x = self.exp2(x,        x_contr2)
    x = self.exp3(x,        x_contr1)
    x = self.exp4(x,        x_in)

    x = self.out(x)

    return x