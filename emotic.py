import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Emotic(nn.Module):
  ''' Emotic Model'''
  def __init__(self, num_context_features, num_body_features):
    n_feature_hidden = 1024
    super(Emotic,self).__init__()
    self.num_context_features = num_context_features
    self.num_body_features = num_body_features
    self.fc1 = nn.Linear((self.num_context_features + num_body_features), n_feature_hidden)
    self.bn1 = nn.BatchNorm1d(n_feature_hidden)
    self.fc2 = nn.Linear(1024, 512)
    self.fc_cat = nn.Linear(512, 26)
    self.fc_cont = nn.Linear(512, 3)
    self.relu = nn.ReLU()

    
  def forward(self, x_context, x_body):
    context_features = x_context.view(-1, self.num_context_features)
    body_features = x_body.view(-1, self.num_body_features)
    fuse_features = torch.cat((context_features, body_features), 1)
    fuse_out = self.fc1(fuse_features)
    fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = F.dropout(fuse_out, p=0.5, training=self.training)  
    fuse_out = self.fc2(fuse_out)
    fuse_out = F.dropout(fuse_out, p=0.5, training=self.training)    
    cat_out = self.fc_cat(fuse_out)
    cont_out = self.fc_cont(fuse_out)
    return cat_out, cont_out
