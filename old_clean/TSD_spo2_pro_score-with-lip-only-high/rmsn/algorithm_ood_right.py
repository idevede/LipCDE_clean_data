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
from rmsn.linear_sequential import LinearSequentialLayer
from rmsn.fft_conv import fft_conv, fft_conv_high_pass , FFTConv1d # 
# We acknowledge the use of tutorials at https://github.com/patrick-kidger/torchcde that served as a skeleton

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# if device == 'cuda':
#     cudnn.benchmark = True
#     #torch.cuda.manual_seed(args.seed)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


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
        self.confounder = 5
        self.sample = 50
        self.relu = nn.ReLU(inplace=True).to(device)
        self.fc1 = nn.Linear(31*hidden_channels, hidden_channels).to(device)
        self.fc2 = nn.Linear(hidden_channels, dim_out).to(device)
        self.fc3 = nn.Linear(19*hidden_channels, hidden_channels).to(device)
        self.fc4 = nn.Linear(hidden_channels, 29).to(device)
        self.fc5 = nn.Linear(4*hidden_channels, hidden_channels).to(device)
        self.fc6 = nn.Linear(hidden_channels, 29).to(device)
        self.fc = nn.Sequential(norm(dim_out), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim_out, 1), nn.Sigmoid()).to(device)
        self.encoder = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers = 2, batch_first=True).to(device)
        self.linear = nn.Linear(hidden_size, self.confounder).to(device)
        self.linear2 = nn.Linear(self.sample*self.sample, input_channels*input_size).to(device)
        self.layer_norm = nn.LayerNorm([self.input_time,self.input_var]).to(device)
        self.linear = LinearSequentialLayer(
            hidden_size,
            self.confounder,
            self.confounder,
            dropout_prob=0.1,
            k_lipschitz=1,
            num_layers=1,
            batch_norm=True).to(device)


    def forward(self, x):
        
        # 0513之前的版本
        # lb = LipschitzBound(x.unsqueeze(0).shape, padding=1, sample=self.sample, backend='torch', cuda=False)
        # sv_bound, x_area = lb.compute(x.unsqueeze(0)) #x_area: x[1]*(50*50)
        # x_area = self.linear2(x_area).view(-1,self.input_time,self.input_var)
        # x_area = self.layer_norm(x_area)
        # hid_x,_ = self.encoder(x_area+x)
        
        # out = self.linear(hid_x)
        # 现在：
        b,l,c = x.shape
        signal = x.view(b,c,l)

        fft_conv = FFTConv1d(self.input_var, self.input_var, 3, bias=True)#.to(device)
        # fft_conv.weight = self.kernel
        # fft_conv.bias = self.bias
        
        out, kernel = fft_conv(signal)
        # lb = LipschitzBound(kernel.shape, padding=1, sample=self.sample, backend='torch', cuda=True)
        # sv_bound = lb.compute(kernel) #x_area: x[1]*(50*50)
        #x_area = self.linear2(out).view(-1,self.input_time,self.input_var)
        x_area = out.view(-1,self.input_time,self.input_var)
        x_area = self.layer_norm(x_area)
        hid_x,_ = self.encoder(x_area+x)
        
        out = self.linear(hid_x)


        #print(out)
        return out



class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,device = 'cpu'):
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
        self.W = torch.nn.Parameter(torch.Tensor(input_channels)).to(device)
        self.W.data.fill_(1)
        self.layer_norm = nn.LayerNorm([hidden_channels]).to(device)

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
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        #print(t)
        z = self.linear1(z)
        z = self.layer_norm(z)
        z = self.elu(z)
        z = self.linear2(z)
        #z = z.tanh()
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        z = torch.matmul(z, torch.diag(self.W))
        #print(self.l1_reg())
        return z


# Next, we need to package CDEFunc up into a model that computes the integral.
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, covariates, hidden_channels, diff_channel=116,device='cpu'):
        super(NeuralCDE, self).__init__()

        self.device = device

        self.func = CDEFunc(covariates, hidden_channels,device = device).to(device)
        self.func2 = CDEFunc(covariates, hidden_channels,device = device).to(device)
        self.initial = torch.nn.Linear(covariates, hidden_channels).to(device)
        #self.readout = torch.nn.Linear(hidden_channels, 1)
        self.readout = torch.nn.Linear(hidden_channels, 2).to(device)
        self.diff = Diffusion(input_channels, diff_channel, covariates, hidden_channels,device = device).to(device)#(1x8004 and 2484x36) 1x8004 = 116*69 2848=36*89
        self.readin = torch.nn.Linear(9, 1).to(device)
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=input_channels, batch_first=True,bidirectional=False).to(device)
        self.lstm2 = torch.nn.LSTM(input_size=input_channels, hidden_size=input_channels, batch_first=True,bidirectional=True).to(device) 
        #torch.nn.utils.rnn.pack_padded_sequence()
        self.readout_lstm = torch.nn.Linear(input_channels*2, 1).to(device)
        self.emb = nn.Linear(1,2).to(device)

    def forward(self, data, sequence_length= None,  training_diffusion=False):

        #data = self.readin(data) #[1, 32, 29]

        #data = data.permute(2, 0, 1)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data) #[1,31,116]

        X = torchcde.CubicSpline(coeffs)

        z0 = self.initial(X.evaluate(0.0))

        z1 = self.initial(X.evaluate(1.0))
        if not training_diffusion:
            # Actually solve the CDE.
            step_size = (X.grid_points[1:] - X.grid_points[:-1]).min()
            z_hat = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.grid_points).to(self.device)#,options=dict(step_size=1))
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
            pred_y = self.readout(z_hat+0.1*self.diff(data))
            #pred_y = self.readout(z_hat)
            #pred_y = self.readout(z_hat)
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


def train(model, train_X, train_y, test_X, test_y, iterations=1000, l1_reg=0.0001):
    optimizer = torch.optim.Adam(model.parameters())

    optimizer_G = torch.optim.Adam(model.parameters(), lr=0.01)
    real_label = 0
    fake_label = 1
    criterion = nn.BCELoss()


    # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
    #     train_X
    # )

    l2_reg = 0.001
    # l1_reg = 0.0001
    # print(iterations)
    train_loss_in = 0
    train_loss_out = 0

    for i in range(iterations):
        #print(i)
        # horizon = 5
        # index = torch.from_numpy(np.random.choice(np.arange(coeffs.shape[1] - batch_time, dtype=np.int64),
        #                                      1, replace=False))
        pred_y = model(train_X)
        loss = F.mse_loss(pred_y, train_y)
        loss = loss + l1_reg * model.func.l1_reg() + l2_reg * model.func.l2_reg()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # train out_data
        label = torch.full(pred_y.size(), real_label)
        #print(label.size())
        #train_coeffs = torch.randn(train_coeffs.size())
        optimizer_G.zero_grad()
        predict_in = model(train_coeffs, training_diffusion=True)
        label = torch.full(predict_in.size(), real_label)
        
        # #print(predict_in.size())
        loss_in = criterion(predict_in.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        loss_in.backward()
        label.fill_(fake_label)

        inputs_out = 2*torch.randn(train_coeffs.size())+train_coeffs #torch.rand_like(inputs)
        predict_out = model(train_coeffs, training_diffusion=True)
        label = torch.full(predict_out.size(), fake_label)
        #print(predict_out)
        loss_out = criterion(predict_out.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        
        loss_out.backward()
        train_loss_out += loss_out.item()
        train_loss_in += loss_in.item()
        optimizer_G.step()




        # if i % 100 == 0:
        #     # print('Iteration: {}   Training loss: {}'.format(i, loss.item()))
        #     plot_trajectories(test_X, test_y, model=model, title=[i, loss])
        #     predictions_NC_SC = predict(model, test_X).squeeze().numpy()
        #     print(np.mean((predictions_NC_SC-Y_test_numpy)**2))
        #     #clear_output(wait=True)


def train_only(model, train_X, train_y, sequence_length=None, active_entries=None, output=None, iterations=1000, l1_reg=0.0001, train_pro_score = []):
    optimizer = torch.optim.Adam(model.parameters())

    optimizer_G = torch.optim.Adam(model.parameters(), lr=0.01)
    real_label = 0
    fake_label = 1
    criterion = nn.BCELoss()


    # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
    #     train_X
    # )

    l2_reg = 0.001
    # l1_reg = 0.0001
    # print(iterations)
    train_loss_in = 0
    train_loss_out = 0

    for i in range(iterations):
        #print(i)
        # horizon = 5
        # index = torch.from_numpy(np.random.choice(np.arange(coeffs.shape[1] - batch_time, dtype=np.int64),
        #                                      1, replace=False))
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
    