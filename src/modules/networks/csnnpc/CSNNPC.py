import snntorch as snn
import torch
import torch.nn as nn

class CSNNPC(nn.Module):
    def __init__(self, num_inputs, num_outputs, population,  betas, spike_grad, num_steps, thresholds):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population = population
        self.betas = betas
        self.spike_grad = spike_grad
        self.num_steps = num_steps
        self.thresholds = thresholds
        self.architecture = f"Conv1d(1, 32, 2) + MaxPool(2) + LIF + Conv1d(32, 128, 2) + MaxPool(2) + LIF + Conv1d(128, 256, 2) + MaxPool(2) + LIF + Linear(768, {self.population}) + LIF"
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.mp1 = nn.MaxPool1d(2)
        self.lif1 = snn.Leaky(beta=self.betas[0], spike_grad=spike_grad, threshold=self.thresholds[0], learn_beta=True, learn_threshold=True)
        self.conv2 = nn.Conv1d(32, 128, 2)
        self.mp2 = nn.MaxPool1d(2)
        self.lif2 = snn.Leaky(beta=self.betas[1], spike_grad=spike_grad, threshold=self.thresholds[1], learn_beta=True, learn_threshold=True)
        self.conv3 = nn.Conv1d(128, 256, 2)
        self.mp3 = nn.MaxPool1d(2)
        self.lif3 = snn.Leaky(beta=self.betas[2], spike_grad=spike_grad, threshold=self.thresholds[2], learn_beta=True, learn_threshold=True)
        self.fc1 = nn.Linear(768, self.population) 
        self.lif4 = snn.Leaky(beta=self.betas[3], spike_grad=spike_grad, threshold=self.thresholds[3], learn_beta=True, learn_threshold=True, output=True)
        
    def forward(self, x):
        """
        Forward pass of the network.
        ------------------------------------------------------
        Args:
            x (torch.Tensor): input tensor
        ------------------------------------------------------
        Returns:
            cur_last_rec (torch.Tensor): tensor with the last current values
            spk_last_rec (torch.Tensor): tensor with the last spike values
            mem_last_rec (torch.Tensor): tensor with the last membrane values
        """
        cur_last_rec = []
        spk_last_rec = []
        mem_last_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        for _ in range(self.num_steps):
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.mp3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc1(spk3.flatten(1))    
            spk4, mem4 = self.lif4(cur4, mem4)
            cur_last_rec.append(cur4)
            spk_last_rec.append(spk4)
            mem_last_rec.append(mem4)
        return torch.stack(cur_last_rec, dim=0), torch.stack(spk_last_rec, dim=0), torch.stack(mem_last_rec, dim=0)
        
    def get_architecture(self):
        """
        Get the architecture of the network.
        ------------------------------------------------------
        Returns:
            architecture (str): string with the architecture
        """
        return self.architecture
    
    def get_parameters(self):
        """
        Get the parameters of the network.
        ------------------------------------------------------
        Returns:
            parameters (list): list with the parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_num_params(self):
        """
        Get the number of parameters of the network.
        ------------------------------------------------------
        Returns:
            num_params (int): number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)