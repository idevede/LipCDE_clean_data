import pandas as pd
import numpy as np
import math
import torch
import torchcde
import torch.nn as nn
from torch.nn import functional as F
from IPython.display import clear_output
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    def __init__(self, input_channels, hidden_channels,dim_out=29):
        super(Diffusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(31*hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, dim_out)
        self.fc3 = nn.Linear(19*hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, 29)
        self.fc5 = nn.Linear(4*hidden_channels, hidden_channels)
        self.fc6 = nn.Linear(hidden_channels, 29)
        self.fc = nn.Sequential(norm(dim_out), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim_out, 1), nn.Sigmoid())

    def forward(self, x):
        l,h,w = x.size()
        #print(l,h,w)
        if h == 19:
            out = self.relu(self.fc3(x.reshape(l,h*w)))
            out = self.fc4(out)
        elif h == 4:
            out = self.relu(self.fc5(x.reshape(l,h*w)))
            out = self.fc6(out)
        else:
            out = self.relu(self.fc1(x.reshape(l,h*w)))
            out = self.fc2(out)

        out = torch.sigmoid(out)
        #print(out)
        return out



class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(
            hidden_channels, input_channels * hidden_channels
        )
        self.elu = torch.nn.ELU(inplace=True)
        self.W = torch.nn.Parameter(torch.Tensor(input_channels))
        self.W.data.fill_(1)

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
        z = self.linear1(z)
        z = self.elu(z)
        z = self.linear2(z)
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        z = torch.matmul(z, torch.diag(self.W))
        return z


# Next, we need to package CDEFunc up into a model that computes the integral.
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, diff_channel=116):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(9, hidden_channels)
        self.func2 = CDEFunc(9, hidden_channels)
        self.initial = torch.nn.Linear(9, hidden_channels)
        #self.readout = torch.nn.Linear(hidden_channels, 1)
        self.readout = torch.nn.Linear(hidden_channels, 1)
        # self.diff = Diffusion(116, 116)#(1x8004 and 2484x36) 1x8004 = 116*69 2848=36*89
        self.diff = Diffusion(diff_channel, diff_channel)#(1x8004 and 2484x36) 1x8004 = 116*69 2848=36*89
        self.readin = torch.nn.Linear(9, 1)
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=29, batch_first=True,bidirectional=False) 
        #torch.nn.utils.rnn.pack_padded_sequence()
        self.emb = nn.Linear(1,2)

    def forward(self, data, sequence_length= None,  training_diffusion=False):

        #data = self.readin(data) #[1, 32, 29]

        #data = data.permute(2, 0, 1)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(data) #[1,31,116]

        X = torchcde.CubicSpline(coeffs)

        z0 = self.initial(X.evaluate(0.0))

        z1 = self.initial(X.evaluate(1.0))
        if not training_diffusion:
            # Actually solve the CDE.
            z_hat = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.grid_points)

            z_hat1 = torchcde.cdeint(X=X, z0=z1, func=self.func2, t=X.grid_points)
            #print(self.diff(coeffs).mean())
            # pred_y = self.readout(0.67*(z_hat+0.5*z_hat1).squeeze(0)).unsqueeze(0) + 0.01*self.diff(coeffs).mean()#.unsqueeze(0)
            pred_y = self.readout(z_hat)
            #print(pred_y.size())
            pred_y_2 = pred_y#pred_y.view(pred_y.shape[1],pred_y.shape[2],1)
            pred_y_2 = self.emb(pred_y_2)
            embed_input_x_packed = pack_padded_sequence(pred_y_2, sequence_length, batch_first=True, enforce_sorted=False)
            encoder_outputs_packed, _ = self.lstm(embed_input_x_packed)
            encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        else:
            #z_hat1 = torchcde.cdeint(X=X, z0=z1, func=self.func2, t=X.grid_points)
            pred_y = self.diff(coeffs) #self.readout((z_hat1).squeeze(0)).unsqueeze(0)
        

        #pred_y = self.readout(0.67*(z_hat+0.5*z_hat1).squeeze(0)).unsqueeze(0)
        #pred_y = self.readout((z_hat).squeeze(0)).unsqueeze(0)
        #pred_y = pred_y.squeeze(0)
        
        if not training_diffusion:
            return torch.mean(pred_y, dim = 1), torch.mean(encoder_outputs, dim = -1).unsqueeze(2) #pred_y: [1,32,29]
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


def train_only(model, train_X, train_y, sequence_length=None, active_entries=None, output=None, iterations=1000, l1_reg=0.0001):
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
        loss = F.mse_loss(encoder_outputs* active_entries,  output*active_entries)
        loss = loss + l1_reg * model.func.l1_reg() + l2_reg * model.func.l2_reg()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # train out_data
        # label = torch.full(pred_y.size(), real_label)
        # #print(label.size())
        # #train_coeffs = torch.randn(train_coeffs.size())
        # optimizer_G.zero_grad()
        # predict_in = model(train_X, training_diffusion=True)
        # label = torch.full(predict_in.size(), real_label)
        
        # # #print(predict_in.size())
        # loss_in = criterion(predict_in.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        # loss_in.backward()
        # label.fill_(fake_label)

        # inputs_out = 2*torch.randn(train_X.size())+train_X #torch.rand_like(inputs)
        # predict_out = model(train_X, training_diffusion=True)
        # label = torch.full(predict_out.size(), fake_label)
        # #print(predict_out)
        # loss_out = criterion(predict_out.type('torch.DoubleTensor'), label.type('torch.DoubleTensor'))
        
        # loss_out.backward()
        # train_loss_out += loss_out.item()
        # train_loss_in += loss_in.item()
        # optimizer_G.step()




        # if i % 100 == 0:
        #     # print('Iteration: {}   Training loss: {}'.format(i, loss.item()))
        #     plot_trajectories(test_X, test_y, model=model, title=[i, loss])
        #     predictions_NC_SC = predict(model, test_X).squeeze().numpy()
        #     print(np.mean((predictions_NC_SC-Y_test_numpy)**2))
        #     #clear_output(wait=True)
