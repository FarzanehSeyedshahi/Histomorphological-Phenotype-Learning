import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention MIL
class AttentionMIL(nn.Module):
    def __init__(self, hidden_dim, att_dim):
        super(AttentionMIL, self).__init__()
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.K = 1 # dimension of the attention vector

        
        # Attention Mechanism for scoring instances
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.att_dim),
            nn.Tanh(),
            nn.Linear(self.att_dim, self.K)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        # attention scores
        A = self.attention(x)
        A = A.transpose(1, 0)
        A = F.softmax(A, dim=1)
        # weighted instances
        C = torch.mm(A, x)
        
        # classification
        Y_prob = self.classifier(C)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A, C
        