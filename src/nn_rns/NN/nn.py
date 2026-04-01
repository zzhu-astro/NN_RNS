import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from .. import eos


_ASSET_DIR = Path(__file__).resolve().parent


def _asset_path(name: str) -> Path:
  return _ASSET_DIR / name



class causalCNN_static(nn.Module):
    """
    causalCNN_static:
      - The neural network architecture for static neutron star
    """

    def __init__(self, len_kernel, num_quantities, index_pc_start, index_pc_end):
        super(causalCNN_static, self).__init__()

        # index_pc_end must be the last elements, or the element after index_pc_end is nonsense
        self.index_pc_start = index_pc_start
        self.index_pc_end = index_pc_end
        self.len_output = index_pc_end - index_pc_start
        self.len_kernel = len_kernel
        
        # first layer (batch_size, in_channels, eos_size, 2) -> (batch_size, out_channels, eos_size, 1), correlate len_kernel elements
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(64)

        # second layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^2 elements
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel)
        self.bn_2 = nn.BatchNorm2d(64)

        # third layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^3 elements
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**2)
        self.bn_3 = nn.BatchNorm2d(64)

        # forth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^4 elements
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**3)
        self.bn_4 = nn.BatchNorm2d(64)

        # fifth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^5 elements
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**4)
        self.bn_5 = nn.BatchNorm2d(64)

        # last layer (batch_size, channels, eos_size, 1) -> (batch_size, num_quantities, len_output, 1)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=num_quantities, kernel_size=(self.index_pc_start+1, 1), stride=1, padding=0)
        self.bn_output = nn.BatchNorm2d(num_quantities)

    def forward(self, x):
        #self.conv_1.weight.data = torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
        #self.conv_1.bias.data = torch.tensor([0.0], dtype=torch.float32)
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # print("Input shape: ", x.shape)

        # first layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("First layer output: ", x.shape)
        
        # second layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel, 0))
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("second layer output: ", x.shape)

        # third layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**2, 0))
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("third layer output: ", x.shape)

        # forth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**3, 0))
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("forth layer output: ", x.shape)

        # fifth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**4, 0))
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = nn.ReLU()(x)
        # print("fifth layer output: ", x.shape)

        # Last layer
        # x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_output(x)
        # x = self.bn_output(x)
        # x = nn.ReLU()(x)
        # x = x[:, :, self.index_pc_start:self.index_pc_end,:]
        # print("last layer output: ", x.shape)
        # print(x)

        return x
    



class causalCNN_kepler(nn.Module):
    """
    causalCNN_kepler:
      - The neural network architecture for Keplerian-rotating neutron star
    """

    def __init__(self, len_kernel, num_quantities, index_pc_start, index_pc_end, h_plus_cutoff, h_minus_cutoff):
        super(causalCNN_kepler, self).__init__()

        # index_pc_end must be the last elements, or the element after index_pc_end is nonsense
        self.index_pc_start = index_pc_start
        self.index_pc_end = index_pc_end
        self.len_output = index_pc_end - index_pc_start
        self.len_kernel = len_kernel
        self.h_plus_cutoff = h_plus_cutoff
        self.h_minus_cutoff = h_minus_cutoff
        
        # first layer (batch_size, in_channels, eos_size, 2) -> (batch_size, out_channels, eos_size, 1), correlate len_kernel elements
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(64)

        # second layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^2 elements
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel)
        self.bn_2 = nn.BatchNorm2d(64)

        # third layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^3 elements
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**2)
        self.bn_3 = nn.BatchNorm2d(64)

        # forth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^4 elements
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**3)
        self.bn_4 = nn.BatchNorm2d(64)

        # fifth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^5 elements
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**4)
        self.bn_5 = nn.BatchNorm2d(64)

        # last layer (batch_size, channels, eos_size, 1) -> (batch_size, num_quantities, len_output, 1)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=num_quantities, kernel_size=(self.index_pc_start+1, 1), stride=1, padding=0)
        self.bn_output = nn.BatchNorm2d(num_quantities)

    def forward(self, x):
        #self.conv_1.weight.data = torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
        #self.conv_1.bias.data = torch.tensor([0.0], dtype=torch.float32)
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # print("Input shape: ", x.shape)

        # first layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("First layer output: ", x.shape)
        
        # second layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel, 0))
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("second layer output: ", x.shape)

        # third layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**2, 0))
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("third layer output: ", x.shape)

        # forth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**3, 0))
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = nn.ReLU()(x)  # apply ReLU activation
        # print("forth layer output: ", x.shape)

        # fifth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**4, 0))
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = nn.ReLU()(x)
        # print("fifth layer output: ", x.shape)

        # Last layer
        # x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_output(x)
        # x = self.bn_output(x)
        # x = nn.ReLU()(x)
        # x = x[:, :, self.index_pc_start:self.index_pc_end,:]
        # print("last layer output: ", x.shape)
        # print(x)

        # special treatment for h_plus and h_minus
        h_plus = x[:, 8:9, :, :]
        h_minus = x[:, 9:10, :, :]
        
        h_plus_clipped = torch.maximum(h_plus, self.h_plus_cutoff)
        h_minus_clipped = torch.maximum(h_minus, self.h_minus_cutoff)
        
        # Reconstruct the output tensor
        x = torch.cat([x[:, :8, :, :], h_plus_clipped, h_minus_clipped, x[:, 10:, :, :]], dim=1)

        return x



class causalCNN_rotate(nn.Module):
    """
    causalCNN_rotate:
      - The neural network architecture for rotating neutron star
    """

    def __init__(self, len_kernel, len_channel, num_quantities, index_pc_start, index_pc_end, h_plus_cutoff, h_minus_cutoff):
        super(causalCNN_rotate, self).__init__()

        # index_pc_end must be the last elements, or the element after index_pc_end is nonsense
        self.index_pc_start = index_pc_start
        self.index_pc_end = index_pc_end
        self.len_output = index_pc_end - index_pc_start
        self.len_kernel = len_kernel
        self.len_channel = len_channel
        self.h_plus_cutoff = h_plus_cutoff
        self.h_minus_cutoff = h_minus_cutoff
        
        # first layer (batch_size, in_channels, eos_size, 2) -> (batch_size, out_channels, eos_size, 1), correlate len_kernel elements
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=len_channel, kernel_size=(len_kernel, 2), stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(len_channel)
        # self.dropout_1 = nn.Dropout2d(0.1)

        # second layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^2 elements
        self.conv_2 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel)
        self.bn_2 = nn.BatchNorm2d(len_channel)
        # self.dropout_2 = nn.Dropout2d(0.1)

        # third layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^3 elements
        self.conv_3 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**2)
        self.bn_3 = nn.BatchNorm2d(len_channel)
        # self.dropout_3 = nn.Dropout2d(0.1)

        # forth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^4 elements
        self.conv_4 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**3)
        self.bn_4 = nn.BatchNorm2d(len_channel)
        # self.dropout_4 = nn.Dropout2d(0.1)

        # fifth layer (batch_size, channels, eos_size, 1) -> (batch_size, channels, eos_size, 1), correlate len_kernel^5 elements
        self.conv_5 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=len_kernel**4)
        self.bn_5 = nn.BatchNorm2d(len_channel)
        # self.dropout_5 = nn.Dropout2d(0.1)

        # --- Refinement Layers ---
        # sixth layer
        self.conv_6 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=1)
        self.bn_6 = nn.BatchNorm2d(len_channel)
        # self.dropout_6 = nn.Dropout2d(0.1)

        # seventh layer
        self.conv_7 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=1)
        self.bn_7 = nn.BatchNorm2d(len_channel)
        # self.dropout_7 = nn.Dropout2d(0.1)

        # eighth layer
        self.conv_8 = nn.Conv2d(in_channels=len_channel, out_channels=len_channel, kernel_size=(len_kernel, 1), stride=1, padding=0, dilation=1)
        self.bn_8 = nn.BatchNorm2d(len_channel)
        # self.dropout_8 = nn.Dropout2d(0.1)

        # last layer (batch_size, channels, eos_size, 1) -> (batch_size, num_quantities, len_output, 1)
        self.conv_output = nn.Conv2d(in_channels=len_channel, out_channels=num_quantities, kernel_size=(self.index_pc_start+1, 1), stride=1, padding=0)
        # self.bn_output = nn.BatchNorm2d(num_quantities)

    def forward(self, x):
        #self.conv_1.weight.data = torch.tensor([[[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
        #self.conv_1.bias.data = torch.tensor([0.0], dtype=torch.float32)
        # x = x.unsqueeze(0)
        # x = x.unsqueeze(0)
        # print("Input shape: ", x.shape)

        # first layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = nn.ReLU()(x)
        # x = self.dropout_1(x)
        
        # second layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel, 0))
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = nn.ReLU()(x)
        # x = self.dropout_2(x)

        # third layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**2, 0))
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = nn.ReLU()(x)
        # x = self.dropout_3(x)

        # forth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**3, 0))
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = nn.ReLU()(x)
        # x = self.dropout_4(x)

        # fifth layer of convolution
        x = F.pad(x, (0, 0, (self.len_kernel - 1) * self.len_kernel**4, 0))
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = nn.ReLU()(x)
        # x = self.dropout_5(x)

        # sixth layer
        x = F.pad(x, (0, 0, 2, 0))
        x = self.conv_6(x)
        x = self.bn_6(x)
        x = nn.ReLU()(x)
        # x = self.dropout_6(x)

        # seventh layer
        x = F.pad(x, (0, 0, 2, 0))
        x = self.conv_7(x)
        x = self.bn_7(x)
        x = nn.ReLU()(x)
        # x = self.dropout_7(x)

        # eighth layer
        x = F.pad(x, (0, 0, 2, 0))
        x = self.conv_8(x)
        x = self.bn_8(x)
        x = nn.ReLU()(x)
        # x = self.dropout_8(x)

        # Last layer
        # x = F.pad(x, (0, 0, (self.len_kernel - 1), 0))
        x = self.conv_output(x)
        # x = self.bn_output(x)
        # x = nn.ReLU()(x)
        # x = x[:, :, self.index_pc_start:self.index_pc_end,:]
        # print("last layer output: ", x.shape)
        # print(x)

        # special treatment for h_plus and h_minus
        h_plus = x[:, 8:9, :, :]
        h_minus = x[:, 9:10, :, :]
        
        h_plus_clipped = torch.maximum(h_plus, self.h_plus_cutoff)
        h_minus_clipped = torch.maximum(h_minus, self.h_minus_cutoff)
        
        # Reconstruct the output tensor
        x = torch.cat([x[:, :8, :, :], h_plus_clipped, h_minus_clipped, x[:, 10:, :, :]], dim=1)

        return x




class NN_models():
    """
    NN_models:
      - Load all the neural network models
      - load_eos(EoSTable)
      - rns_eval(EoSTable)
    """

    obs_names = ['M', 'M0', 'R', 'Omega', 'T/W', 'J', 'I', 'Phi_2', 'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'Omega_p', 'r_ratio']
    static_obs_names = ['M', 'M0', 'R', 'Omega_p', 'Z_p']
    r_state_names = ["kepler", "r0.50", "r0.55", "r0.60", "r0.65", "r0.70", "r0.75", "r0.80", "r0.85", "r0.90", "r0.95", "static"]
    log_obs = [ 'J', 'I', 'Phi_2', 'Z_p', 'Z_b']

    def __init__(self, r_ratio=None):
        # loading static model
        self.global_s_mean_vals = torch.from_numpy(np.load(_asset_path('global_mean_vals.npy')))
        self.global_s_std_vals = torch.from_numpy(np.load(_asset_path('global_std_vals.npy')))
        self.model_static = causalCNN_static(3, 5, 87, 127)
        self.model_static.load_state_dict(torch.load(_asset_path("best_causalCNN_static_model.pth"), map_location="cpu"))
        # model.to(device)  # Move to GPU or CPU as needed
        self.model_static.eval()      # Set to evaluation mode

        # loading kepler model
        self.global_k_mean_vals = torch.from_numpy(np.load(_asset_path('global_k_mean_vals.npy')))
        self.global_k_std_vals = torch.from_numpy(np.load(_asset_path('global_k_std_vals.npy')))
        h_plus_cutoff_k = (-self.global_k_mean_vals[8] / self.global_k_std_vals[8])
        h_minus_cutoff_k = (-self.global_k_mean_vals[9] / self.global_k_std_vals[9])
        num_quantities = 14

        self.model_kepler = causalCNN_kepler(3, num_quantities, 87, 127, h_plus_cutoff_k, h_minus_cutoff_k)
        self.model_kepler.load_state_dict(torch.load(_asset_path("best_causalCNN_kepler_model.pth"), map_location="cpu"))
        self.model_kepler.eval()      # Set to evaluation mode

        # loading rotate model
        self.global_r_mean_vals = torch.from_numpy(np.load(_asset_path('global_r_mean_vals.npy')))
        self.global_r_std_vals = torch.from_numpy(np.load(_asset_path('global_r_std_vals.npy')))
        h_plus_cutoff_r = (-self.global_r_mean_vals[8] / self.global_r_std_vals[8])
        h_minus_cutoff_r = (-self.global_r_mean_vals[9] / self.global_r_std_vals[9])

        self.model_rotate = causalCNN_rotate(3, 128, num_quantities, 87, 127, h_plus_cutoff_r, h_minus_cutoff_r)
        self.model_rotate.load_state_dict(torch.load(_asset_path("best_causalCNN_rotate_model.pth"), map_location="cpu"))
        self.model_rotate.eval()      # Set to evaluation mode

        # r_ratio can be a scalar or array-like; values must be within [0.5, 1.0]
        if r_ratio is None:
          r_ratio = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        r_ratio = np.asarray(r_ratio, dtype=np.float32)
        if r_ratio.ndim == 0:
          r_ratio = r_ratio.reshape(1)
        assert np.all((r_ratio >= 0.5) & (r_ratio <= 1.0)), "r_ratio must be within [0.5, 1.0]"
        self.r_ratio = r_ratio

        # set the nb array, this is pre-determined by NN
        self.nb_arr = np.exp(np.linspace(np.log(2.57200000e+35), np.log(1.60000000e+39), 128))[:127]


    def load_eos(self, EoSTable):
        p_arr = EoSTable.p_from_nb(self.nb_arr)

        return (p_arr)
    
    def nn_eval(self, EoSTable):
        # load eos
        self.p_arr = np.asarray(self.load_eos(EoSTable), dtype=np.float32)
        self.logp_nn_s_input = torch.from_numpy(np.log(self.p_arr[np.newaxis, np.newaxis, :, np.newaxis])).to(dtype=torch.float32)
        self.logp_nn_k_input = self.logp_nn_s_input.clone()
        len_eos = self.logp_nn_s_input.shape[2]

        r_ratio_tensor = torch.from_numpy(self.r_ratio).to(dtype=torch.float32).view(-1)
        n_ratio = r_ratio_tensor.shape[0]

        logp_expanded = self.logp_nn_s_input.expand(n_ratio, 1, -1, 1)
        r_ratio_expanded = r_ratio_tensor.view(n_ratio, 1, 1, 1).expand(n_ratio, 1, len_eos, 1)
        self.logp_ratio_nn_r_input = torch.cat([logp_expanded, r_ratio_expanded], dim=-1)

        # central variables
        self.press_c = self.p_arr[87:]
        self.energy_c = EoSTable.e_from_p(self.p_arr[87:])
        self.nb_c = EoSTable.nb_from_p(self.p_arr[87:]) 
        self.eos_mask = EoSTable.in_range[87:]

        # static model evaluation, variables of results (columns): 'M', 'M_0', 'R', 'Omega_p', 'Z_p(Z_b, Z_f)'
        predict_s = self.model_static(self.logp_nn_s_input).detach().numpy()
        self.nn_rns_static = np.empty((predict_s.shape[2], predict_s.shape[1]), dtype=np.float32)
        for iq in range(predict_s.shape[1]):
            self.nn_rns_static[:, iq] = predict_s[0, iq, :, 0] * self.global_s_std_vals[iq].item() + self.global_s_mean_vals[iq].item()

        # kepler model evaluation, variables of results (columns): 'M', 'M_0', 'R', 'Omega', 'T/W', 'C*J/GM_s^2', 'I', 'Phi_2', 'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'r_ratio'
        predict_k = self.model_kepler(self.logp_nn_k_input).detach().numpy()
        self.nn_rns_kepler = np.empty((predict_k.shape[2], predict_k.shape[1]), dtype=np.float32)
        for iq in range(predict_k.shape[1]):
            self.nn_rns_kepler[:, iq] = predict_k[0, iq, :, 0] * self.global_k_std_vals[iq].item() + self.global_k_mean_vals[iq].item()

        # rotate model evaluation, variables of results (columns): 'M', 'M_0', 'R', 'Omega', 'T/W', 'C*J/GM_s^2', 'I', 'Phi_2', 'h_plus', 'h_minus', 'Z_p', 'Z_b', 'Z_f', 'r_ratio'
        predict_r = self.model_rotate(self.logp_ratio_nn_r_input).detach().numpy()
        self.nn_rns_rotate = np.empty((predict_r.shape[0], predict_r.shape[2], predict_r.shape[1]), dtype=np.float32)
        for iq in range(predict_r.shape[1]):
            self.nn_rns_rotate[:, :, iq] = predict_r[:, iq, :, 0] * self.global_r_std_vals[iq].item() + self.global_r_mean_vals[iq].item()