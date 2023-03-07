import pandas as pd
import numpy as np
import math
import torch
import torchcde
import torchsde
import torch.nn as nn
from torch.nn import functional as F
from IPython.display import clear_output
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rmsn.lipschitz_bound2 import LipschitzBound
from rmsn.fft_conv import fft_conv, fft_conv_high_pass, FFTConv1d # 

# We acknowledge the use of tutorials at https://github.com/patrick-kidger/torchcde that served as a skeleton

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# if device == 'cuda':
#     cudnn.benchmark = True
#     #torch.cuda.manual_seed(args.seed)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

class CLipSwish(nn.Module):

    def __init__(self):
        super(CLipSwish, self).__init__()
        self.swish = Swish()

    def forward(self, x):
        x = torch.cat((x, -x), 2)
        return self.swish(x).div_(1.004)


class LipschitzDenseLayer(torch.nn.Module):
    def __init__(self, network, learnable_concat=False, lip_coeff=0.98):
        super(LipschitzDenseLayer, self).__init__()
        self.network = network
        self.lip_coeff = lip_coeff
        self.CLipSwish = CLipSwish()
        

        if learnable_concat:
            self.K1_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
            self.K2_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
        else:
            self.register_buffer("K1_unnormalized", torch.tensor([1.]))
            self.register_buffer("K2_unnormalized", torch.tensor([1.]))

    def get_eta1_eta2(self, beta=0.1):
        eta1 = F.softplus(self.K1_unnormalized) + beta
        eta2 = F.softplus(self.K2_unnormalized) + beta
        divider = torch.sqrt(eta1 ** 2 + eta2 ** 2)

        eta1_normalized = (eta1/divider) * self.lip_coeff
        eta2_normalized = (eta2/divider) * self.lip_coeff
        return eta1_normalized, eta2_normalized

    def forward(self, x):
        out = self.network(x)
        eta1_normalized, eta2_normalized = self.get_eta1_eta2()
        result = torch.cat([x * eta1_normalized, out * eta2_normalized], dim=2)
        return  x , out * eta2_normalized #self.CLipSwish(result)


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]),
                                         torch.Tensor([std / n_units]))
    A_init = sampler.sample((n_units, n_units))[..., 0]
    return A_init



class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

        
def plot_trajectories(X, Y, model, title=[1, 2.1]):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.3))
    fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)

    predictions = predict(model, X)
    axs[0].plot(torch.cat([Y, predictions], dim=-1).squeeze())
    axs[0].set_title("Iteration = %i" % title[0] + ",  " + "Loss = %1.3f" % title[1])
    axs[1].plot(Y.squeeze() - predictions.squeeze())
    axs[1].set_title("Treatment Effect")
    axs[2].plot(X[0, :, :])
    axs[2].set_title("Control trajectories")
    plt.show()

class Diffusion(nn.Module):
    def __init__(self, input_channels, hidden_channels,input_size, hidden_size,dim_out=29,device = 'cpu'):
        super(Diffusion, self).__init__()
        self.input_time = input_channels
        self.input_var = input_size
        self.confounder = input_size
        self.device =device
        self.relu = nn.ReLU(inplace=True).to(device)
        self.encoder = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers = 2, batch_first=True).to(device)
        self.linear = nn.Linear(hidden_size, self.confounder).to(device)
        self.layer_norm = nn.LayerNorm([self.input_time,self.input_var]).to(device)
        self.fft_conv = FFTConv1d(input_size, input_size, 3, bias=True).to(device)


    def forward(self, x):
        
        b,l,c = x.shape
        signal = x.view(b,c,l)

        # fft_conv = self.FFTConv(self.input_var, self.input_var, 3, bias=True)#.to(device)
        # fft_conv.weight = self.kernel
        # fft_conv.bias = self.bias
        
        out, kernel = self.fft_conv(signal)
        # lb = LipschitzBound(kernel.shape, padding=1, sample=self.sample, backend='torch', cuda=True)
        # sv_bound = lb.compute(kernel) #x_area: x[1]*(50*50)
        # x_area = self.linear2(out).view(-1,self.input_time,self.input_var)
        x_area = out.view(-1,self.input_time,self.input_var)
        x_area = self.layer_norm(x_area)
        hid_x,_ = self.encoder(x_area+x)
        
        out = self.linear(hid_x)

        return out



class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,device = 'cpu', gamma =0.01, beta = 0.8, n_units = 128, init_std = 1):
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(
            hidden_channels, input_channels * hidden_channels
        ).to(device)
        self.elu = torch.nn.LeakyReLU(inplace=True).to(device) # ELU
        # self.W = torch.nn.Parameter(torch.Tensor(input_channels)).to(device)
        # self.W.data.fill_(1)
        self.layer_norm = nn.LayerNorm([hidden_channels]).to(device)
        self.layer_norm_2 = nn.LayerNorm([hidden_channels,input_channels]).to(device)

        self.gamma = gamma
        self.beta = beta

        self.tanh = nn.Tanh().to(device)

        self.z = torch.zeros(input_channels).to(device)
        self.C = nn.Parameter(gaussian_init_(input_channels, std=init_std)).to(device)
        self.B = nn.Parameter(gaussian_init_(input_channels, std=init_std)).to(device)
        self.I = torch.eye(input_channels).to(device)
        self.i = 0

    def l2_reg(self):
        """L2 regularization on all parameters"""
        reg = 0.0
        reg += torch.sum(self.linear1.weight ** 2)
        reg += torch.sum(self.linear2.weight ** 2)
        return reg

    def l1_reg(self):
        """L1 regularization on input layer parameters"""
        return torch.sum(torch.abs(self.W))

    # The t argument can be ignored or added specifically if you want your CDE to behave differently at
    # different times.
    # def forward(self, t, z):
    #     # z has shape (batch, hidden_channels)
    #     #print(t)
    #     z = self.linear1(z)
    #     z = self.layer_norm(z)
    #     z = self.elu(z)
    #     z = self.linear2(z)
    #     #z = z.tanh()
    #     # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
    #     # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
    #     z = z.view(z.size(0), self.hidden_channels, self.input_channels)
    #     z = torch.matmul(z, torch.diag(self.W))
    #     #print(self.l1_reg())
    #     print(self.W)
    #     return z

    def forward(self, t, h):    
        """dh/dt as a function of time and h(t)."""
        z = self.linear1(h)
        z = self.layer_norm(z)
        z = self.elu(z)
        z = self.linear2(z)
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        h = z.view(z.size(0), self.hidden_channels, self.input_channels)

        if self.i == 0:
            self.A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                1 - self.beta) * (self.B +
                                  self.B.transpose(1, 0)) - self.gamma * self.I  # 39*39
            self.W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                1 - self.beta) * (self.C +
                                  self.C.transpose(1, 0)) - self.gamma * self.I # 39 *39 
            #self.i = 1
            

        return torch.matmul(
            h, self.A) + self.tanh(torch.matmul(h, self.W) + self.z)

# Next, we need to package CDEFunc up into a model that computes the integral.
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, covariates, hidden_channels, diff_channel=116,device='cuda:0'):
        super(NeuralCDE, self).__init__()

        self.device = device

        self.func = CDEFunc(covariates+2, hidden_channels,device = device).to(device)
        self.func2 = CDEFunc(covariates, hidden_channels,device = device).to(device)
        self.initial = torch.nn.Linear(covariates+2, hidden_channels).to(device)
        self.initial_hidden = torch.nn.Linear(covariates, 2).to(device)
        #self.readout = torch.nn.Linear(hidden_channels, 1)
        self.readout = torch.nn.Linear(hidden_channels, 2).to(device)
        self.diff = Diffusion(input_channels, diff_channel, covariates, hidden_channels,device = device).to(device)#(1x8004 and 2484x36) 1x8004 = 116*69 2848=36*89
        self.LipDense = LipschitzDenseLayer(self.diff, learnable_concat= True).to(device)

        self.readin = torch.nn.Linear(9, 1).to(device)
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=input_channels, batch_first=True,bidirectional=False).to(device)
        self.lstm2 = torch.nn.LSTM(input_size=input_channels, hidden_size=input_channels, batch_first=True,bidirectional=True).to(device) 
        #torch.nn.utils.rnn.pack_padded_sequence()
        self.readout_lstm = torch.nn.Linear(input_channels*2, 1).to(device)
        self.emb = nn.Linear(1,2).to(device)
        self.att = nn.MultiheadAttention(covariates, 1,batch_first=True).to(device)

    def forward(self, data, sequence_length= None,  training_diffusion=False):

        #data = self.readin(data) #[1, 32, 29]

        #data = data.permute(2, 0, 1)
        data_diff = data#.copy()
        data, hidden_data = self.LipDense(data) # data: 1 19 39
        hidden_data = self.initial_hidden(hidden_data)
        data = torch.cat([data,hidden_data], dim = 2)
        #data = data + 0.01*hidden_data
        #data, data_att_weight = self.att(data,data, hidden_data) 

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data) #[1,31,116]

        X = torchcde.CubicSpline(coeffs)

        z0 = self.initial(X.evaluate(0.0))

        #z1 = self.initial(X.evaluate(1.0))
        if not training_diffusion:
            # Actually solve the CDE.
            adjoint_params = tuple(self.func.parameters()) + (coeffs,)
            z_hat = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.grid_points, adjoint_params = adjoint_params).to(self.device)#,options=dict(step_size=1))
            
            #z_hat = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.grid_points)
            #z_hat = torchsde.sdeint(sde = self.func, ts=X, y0=z0)
            #print('finish z_hat')
            #z_hat1 = torchcde.cdeint(X=X, z0=z1, func=self.func2, t=X.grid_points)
            #print(self.diff(coeffs).mean())
            # pred_y = self.readout(0.67*(z_hat+0.5*z_hat1).squeeze(0)).unsqueeze(0) + 0.01*self.diff(coeffs).mean()#.unsqueeze(0)
            #pred_y = self.readout(z_hat)
            #pred_y = self.readout(0.67*(z_hat+0.5*self.diff(data)))
            # new version
            
            #print(z_hat)
            # print(self.diff(data).shape)
            # hidden_data = self.diff(data)
            # #z_hat_att, z_hat_att_weight = self.att(hidden_data,hidden_data,z_hat) 
            # z_hat_att, z_hat_att_weight = self.att(z_hat,z_hat, hidden_data) 
            # pred_y = self.readout(z_hat_att)
            #pred_y = self.readout(z_hat)
            #pred_y = self.readout(z_hat)
            #print(z_hat.shape)
            #print(self.diff(data_diff).shape)
            pred_y = self.readout(z_hat)

            #print(pred_y.size())
            pred_y_2 = pred_y #pred_y.view(pred_y.shape[1],pred_y.shape[2],1)
            #pred_y_2 = self.emb(pred_y_2)
            #print(pred_y_2[0,0,:])
            # print(pred_y_2.shape)
            embed_input_x_packed = pack_padded_sequence(pred_y_2, sequence_length.to("cpu"), batch_first=True, enforce_sorted=False)
            encoder_outputs_packed, hidden = self.lstm(embed_input_x_packed)
            #encoder_outputs_packed, _ = self.lstm2(hidden)
            encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True) #32,29,29
            encoder_outputs_packed, hidden = self.lstm2(encoder_outputs)
            encoder_outputs = self.readout_lstm(encoder_outputs_packed)
            #print(encoder_outputs.shape)

        else:
            #z_hat1 = torchcde.cdeint(X=X, z0=z1, func=self.func2, t=X.grid_points)
            #pred_y = self.diff(coeffs) #self.readout((z_hat1).squeeze(0)).unsqueeze(0)
            pred_y = self.readout(self.diff(data)) #self.readout((z_hat1).squeeze(0)).unsqueeze(0)
            pred_y = torch.sigmoid(pred_y)
        

        #pred_y = self.readout(0.67*(z_hat+0.5*z_hat1).squeeze(0)).unsqueeze(0)
        #pred_y = self.readout((z_hat).squeeze(0)).unsqueeze(0)
        #pred_y = pred_y.squeeze(0)
        
        if not training_diffusion:
            max_length = 29 # max time stapes = sequence_max 
            #return torch.mean(pred_y, dim = 1), torch.mean(encoder_outputs, dim = -1).unsqueeze(2) #pred_y: [1,32,29]
            if encoder_outputs.shape[1] != max_length: #
                #print(encoder_outputs.shape[1])
                tensor_zero = torch.zeros((encoder_outputs.shape[0],max_length-encoder_outputs.shape[1],encoder_outputs.shape[2])).to(self.device)
                encoder_outputs = torch.cat((encoder_outputs,tensor_zero),1)
            #return torch.mean(pred_y, dim = 1), torch.mean(encoder_outputs, dim = -1).unsqueeze(2) #pred_y: [1,32,29]
            return torch.mean(pred_y, dim = 1), encoder_outputs #pred_y: [1,32,29]
        else:
            return pred_y

    def predict(self, data, z0):

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data)
        X = torchcde.CubicSpline(coeffs)

        # z0 = X.evaluate(0.)

        # Actually solve the CDE.
        z_hat = torchcde.cdeint(
            X=X,
            z0=z0,
            func=self.func,
            t=torch.linspace(X.grid_points[0], X.grid_points[-1], 100),
        )

        return z_hat.detach()


def predict(model, test_X, sequence_length):
    full_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
    return model(test_X, sequence_length) #.detach()


def train_only(model, train_X, train_y, sequence_length=None, active_entries=None, output=None, iterations=1000, l1_reg=0.0001, train_pro_score = []):
    optimizer = torch.optim.Adam(model.parameters())

    l2_reg = 0.001
   

    for i in range(iterations):
       
        pred_y , encoder_outputs = model(train_X, sequence_length)

        b, n, s = encoder_outputs.shape
        b1, n1, s1 = output.shape
        if n != n1:
            continue

        #loss = F.mse_loss(pred_y, train_y) + F.mse_loss(encoder_outputs* active_entries - output*active_entries)
        loss = F.mse_loss(encoder_outputs* active_entries*train_pro_score,  output*active_entries*train_pro_score)

        loss = loss + l1_reg * model.func.l1_reg() + l2_reg * model.func.l2_reg()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    