import torch
from torch import nn
import torch.nn.functional as F

############################ helper conv function ##############################
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)


############################# helper deconv function ##############################
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    return nn.Sequential(*layers)


############################# residual block class #################################
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=True)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2
    
############################### GENERATOR MODEL ###############################

class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=32, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        
        # initial convolutional layer given, below
        self.conv1 = conv(1, conv_dim, 25, 4, 11)
        self.conv2 = conv(conv_dim, conv_dim*4, 25, 4, 11) # conv_dim out was *2
        self.conv3 = conv(conv_dim*4, conv_dim*8, 25, 4, 11) # conv_dim out was *4
        self.conv4 = conv(conv_dim*8, conv_dim*16, 25, 4, 11) # conv_dim out was *4

        # 2. Define the resnet part of the generator
        # Residual blocks
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim*16)) # conv_dim out was *4
        # use sequential to create these layers
        self.res_blocks = nn.Sequential(*res_layers)

        # 3. Define the decoder part of the generator
        # two transpose convolutional layers and a third that looks a lot like the initial conv layer
        self.deconv1 = deconv(conv_dim*16, conv_dim*8, 24, 4, 10) # conv_dim out was *2
        self.deconv2 = deconv(conv_dim*8, conv_dim*4, 24, 4, 10) # conv_dim out was *2
        self.deconv3 = deconv(conv_dim*4, conv_dim, 24, 4, 10) # conv_dim out was *1
        # no batch norm on last layer
        self.deconv4 = deconv(conv_dim, 1, 24, 4, 10, batch_norm=False)
        
        ### Initialize weights ###
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(module.weight.data)

    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary

        out = F.relu(self.conv1(x))
        #print (out.shape)
        out = F.relu(self.conv2(out))
        #print (out.shape)
        out = F.relu(self.conv3(out))
        #print (out.shape)        
        out = F.relu(self.conv4(out))
        #print (out.shape)
        out = self.res_blocks(out)
        #print (out.shape)
        out = F.relu(self.deconv1(out))
        #print (out.shape)
        out = F.relu(self.deconv2(out))
        #print (out.shape)
        out = F.relu(self.deconv3(out))
        # tanh applied to last layer
        out = F.tanh(self.deconv4(out))
        #print (out.shape)
        return out
    
class waveganGenerator(nn.Module):
    
    def __init__(self, d):  # d = model_size
        
        super(waveganGenerator, self).__init__()
        self.d = d
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size = 25, stride = 4, padding = 11)
        self.conv2 = nn.Conv1d(16, 2*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv3 = nn.Conv1d(2*d, 4*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv4 = nn.Conv1d(4*d, 16*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv5 = nn.Conv1d(16*d, 32*d, kernel_size = 25, stride = 4, padding = 11)
              
        ### Initialize weights ###
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
        
        #self.dropout = nn.Dropout(0.3)
    
    def forward(self, z):                       # z shape: (64, 1, 16384)
        
        #z = z.reshape((-1, 16384, 1))          #(64,16384,1)
        #print(z.shape)
        z = F.relu(self.conv1(z))     #(64, 4, 4096)
        #print(z.shape)
        z = F.relu(self.conv2(z))     #(64, 16, 1024)
        #print(z.shape)
        z = F.relu(self.conv3(z))     #(64, 2d, 256)
        #print(z.shape)
        z = F.relu(self.conv4(z))     #(64, 8d, 64)
        #print(z.shape)    
        output = F.relu(self.conv5(z))#(64, 32d, 16)       
        #print(output.shape)
        output = output.reshape((-1, 1, 16384)) #(64, 1, 16384)
        #print(output.shape)
        return output



################### PHASE SHUFFLE MODEL FOR DISCRIMINATOR #####################
############# (taken from:  https://github.com/jtcramer/wavegan) ##############
class PhaseShuffle(nn.Module):
    '''
    Performs phase shuffling (to be used by Discriminator ONLY) by: 
       -Shifting feature axis of a 3D tensor by a random integer in [-n, n] 
       -Performing reflection padding where necessary
    '''
    def __init__(self, shift_factor):
        
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        
        
    def forward(self, x):    # x shape: (64, 1, 16384)
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x
        
        # k_list: list of batch_size shift factors, randomly generated uniformly between [-shift_factor, shift_factor]
        k_list = torch.Tensor(x.shape[0]).random_(0, 2*self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)
        
        # k_map: dict containing {each shift factor : list of batch indices with that shift factor} 
        # e.g. if shift_factor = 1 & batch_size = 64, k_map = {-1:[0,2,30,...52], 0:[1,5,...,60], 1:[2,3,4,...,63]}
        k_map = {}  
        for sample_idx, k in enumerate(k_list):
            k = int(k) 
            if k not in k_map:
                k_map[k] = []
            
            k_map[k].append(sample_idx)
            
        shuffled_x = x.clone()
        
        for k, sample_idxs in k_map.items():
            if k > 0:   # Remove the last k values & insert k left-paddings 
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., :-k], 
                                                pad = (k, 0), 
                                                mode = 'reflect')
            
            else:       # 1. Remove the first k values & 2. Insert k right-paddings 
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., abs(k):], 
                                                  pad = (0, abs(k)), 
                                                  mode = 'reflect')
                    
        assert shuffled_x.shape == x.shape, "{}, {}".format(shuffled_x.shape, x.shape)
        
        return shuffled_x
        
     

############################# DISCRIMINATOR MODEL #############################      
class Discriminator(nn.Module):
    
    def __init__(self, d, shift_factor = 2, alpha = 0.2):
        
        super(Discriminator, self).__init__()
        self.d = d
        self.alpha = alpha
        
        self.conv1 = nn.Conv1d(1, d, kernel_size = 25, stride = 4, padding = 11)
        self.conv2 = nn.Conv1d(d, 2*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv3 = nn.Conv1d(2*d, 4*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv4 = nn.Conv1d(4*d, 8*d, kernel_size = 25, stride = 4, padding = 11)
        self.conv5 = nn.Conv1d(8*d, 16*d, kernel_size = 25, stride = 4, padding = 11)
        self.dense = nn.Linear(256*d, 1)
        self.phase_shuffle = PhaseShuffle(shift_factor)
        
        ### Initialize weights ###
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):                                # x shape: (64, 1, 16384)

        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha) #(64, d, 4096)
        x = self.dropout(x)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha) #(64, 2d, 1024)
        x = self.dropout(x)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha) #(64, 4d, 256)
        x = self.dropout(x)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha) #(64, 8d, 64)
        x = self.dropout(x)
        x = self.phase_shuffle(x)
        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha) #(64, 16d, 16)
        x = self.dropout(x)
        x = x.reshape((-1, 256*self.d))                            #(64, 256d)
        output = self.dense(x)                                     #(64, 1)
        
        return output
