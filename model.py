import torch
import torch.nn as nn


class EdgeNetwork(nn.Module):
    def __init__(self, in_features, outfeatures, first=False):
        super().__init__()

        self.first = first

        midfeatures = int((2*in_features+outfeatures)/2)
        self.net = nn.Sequential(
                nn.Linear(2*in_features+1,midfeatures),
                nn.ReLU(),
                nn.Linear(midfeatures,outfeatures),
                nn.Tanh())        
            
        
    def forward(self, inp_node, inp_edge):
        
        B, n, _ = inp_node.shape

        ind1 = torch.arange(n).view(-1,1).repeat(1,n-1).view(-1)
        m1 = inp_node[:,ind1,:]

        ind2 = torch.nonzero(torch.arange(n*n)%(n+1) != 0).view(-1)
        m2 = inp_node.repeat(1, n, 1)[:,ind2,:]

        block = torch.cat([m1,m2],dim=2)
        block = block.view(B, n, n-1, block.shape[-1])
        block = torch.cat([block, inp_edge.reshape(B, n, n-1, 1)], dim=3)
        value = self.net(block)
                
        message_sum = torch.sum(value,dim=2)
        
        return message_sum


class NodeNetwork(nn.Module):
    def __init__(self, original_in_features, in_features, out_features):
        
        super(NodeNetwork, self).__init__()
        
        out_features = int(out_features/2)
        mid_features = int((original_in_features+in_features+out_features)/2)
        
        self.edgenet = EdgeNetwork(original_in_features+in_features, out_features)
        self.node_layer1 = nn.Sequential(
            nn.Linear(original_in_features+in_features, mid_features, bias=True),
            nn.ReLU(),
            nn.Linear(mid_features, out_features, bias=True),
            nn.Tanh()
        )
        self.node_layer2 = nn.Sequential(
            nn.Linear(out_features, out_features, bias=True),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=True),
            nn.Tanh()
        )

    def forward(self, inp_node, inp_edge):

        B, N, _ = inp_node.shape

        message_sum = self.edgenet(inp_node, inp_edge)
        
        out1 = self.node_layer1(inp_node) 
        out2 = self.node_layer2(message_sum)

        out = torch.cat([out1, out2],dim=2)
        out = out / torch.norm(out, p='fro', dim=2, keepdim=True)
        
        return out
    
    
class MyModel(nn.Module):
    def __init__(self, in_features, feats, correction_layers, num_wp=1):
        super(MyModel, self).__init__()
  
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        self.flav_ordered = torch.FloatTensor([5,4,0,15])
        self.embedding = nn.Embedding(4, 3)
        
        self.gnn_layers = nn.ModuleList([])
        self.gnn_layers.append(NodeNetwork(in_features, 0, feats[0]))
        for i in range(1, len(feats)):
            self.gnn_layers.append(NodeNetwork(in_features, feats[i-1], feats[i]))

        self.n_layers = len(self.gnn_layers)
                
        f_correction_layers = []
        f_correction_layers.append(nn.Linear(in_features+feats[-1],correction_layers[0]))
        f_correction_layers.append(nn.ReLU())
            
        for hidden_i in range(1,len(correction_layers)):
            f_correction_layers.append(nn.Linear(correction_layers[hidden_i-1],correction_layers[hidden_i]))
            f_correction_layers.append(nn.ReLU())
            
        f_correction_layers.append(nn.Linear(correction_layers[-1],num_wp))
        if num_wp == 1:
            f_correction_layers.append(nn.Sigmoid())
        else:
            f_correction_layers.append(nn.Softmax(dim=2))
        
        self.correction = nn.Sequential( *f_correction_layers )
            

    def calculate_deltaR(self, eta1, phi1, eta2, phi2):
        deta = eta1-eta2
        dphi = phi1-phi2
        
        dphi[torch.nonzero(dphi >= self.pi)] -= 2*self.pi
        dphi[torch.nonzero(dphi < -self.pi)] += 2*self.pi

        return torch.sqrt(deta*deta + dphi*dphi);


    def forward(self, node_feat):
        
        '''
            node_feat: shape (1, num_nodes, num_nodefeat)
            *only batch size = 1 is accepted*
        '''
                
        # extract the inputs
        inp_flavind = torch.nonzero(self.flav_ordered==node_feat[:,:,0].unsqueeze(2))[:,-1].reshape(-1,node_feat.shape[1])
        
        _, n, _ = node_feat.shape
        eta, phi = node_feat[0,:,2], node_feat[0,:,3] 
        ind1 = torch.arange(n).view(-1,1).repeat(1,n-1).view(-1)        
        ind2 = torch.nonzero(torch.arange(n*n)%(n+1) != 0).view(-1)%n
        dr = self.calculate_deltaR(eta[ind1], phi[ind1], eta[ind2], phi[ind2])
                
        inp_node = node_feat[:,:,1:]
        inp_edge = dr.reshape(1, -1)  # the network was built with edge feat shape (B, N^2-N)
        
        
        # normalisation        
        inp_node = torch.cat([inp_node, self.embedding(inp_flavind)], dim=2)

        x_node, x_edge = inp_node, inp_edge
        
                
        for layer_i in range(self.n_layers):
            x_node = self.gnn_layers[layer_i](x_node, inp_edge)
            x_node = torch.cat((inp_node, x_node),dim=-1)
            
        effs = self.correction(x_node)         
        return effs
