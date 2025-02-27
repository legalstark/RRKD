import torch.nn as nn
import torch




class CNN8_S(nn.Module):                                                
    def __init__(self, train_shape, category):
        super(CNN8_S, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
       '''                                                     
        self.layer = nn.Sequential(
            nn.Conv2d(1, 8, (5, 1), (1, 1), (2, 0)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((5,1)), 

            nn.Conv2d(8, 16, (5, 1), (1, 1), (2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((5,1)),

            nn.Conv2d(16, 32, (5, 1), (1, 1), (2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((5,1)),
            )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(32*train_shape[-1], category) 


    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


