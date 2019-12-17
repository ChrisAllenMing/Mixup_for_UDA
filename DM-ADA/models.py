import torch, utils
import torch.nn as nn
from torch.autograd import Variable


"""
Generator network
"""
class _netG(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netG, self).__init__()
        
        self.ndim = opt.ndf*4
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ndim + self.nz + nclasses + 1, self.ngf*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, self.nz, 1, 1).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        output = self.main(torch.cat((input, noisev),1))

        return output

"""
Discriminator network
"""
class _netD(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD, self).__init__()
        
        self.ndf = opt.ndf
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 3, 1, 1),            
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf, self.ndf*2, 3, 1, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),        

            nn.Conv2d(self.ndf*2, self.ndf*4, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(self.ndf*4, self.ndf*2, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4)           
        )

        self.classifier_c = nn.Sequential(nn.Linear(self.ndf*2, nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(self.ndf, 1),
        						nn.Sigmoid())
        self.classifier_t = nn.Sequential(nn.Linear(self.ndf*2, self.ndf))

    def forward(self, input):       
        output = self.feature(input)
        output_c = self.classifier_c(output.view(-1, self.ndf * 2))
        output_t = self.classifier_t(output.view(-1, self.ndf * 2))
        output_s = self.classifier_s(output_t)
        output_s = output_s.view(-1)
        return output_s, output_c, output_t

"""
Feature extraction network
"""
class _netF(nn.Module):
    def __init__(self, opt):
        super(_netF, self).__init__()

        
        self.ndf = opt.ndf
        self.nz = opt.nz
        self.gpu = opt.gpu

        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(self.ndf, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
                    
            nn.Conv2d(self.ndf, self.ndf*2, 5, 1,0),
            nn.ReLU(inplace=True)
        )

        self.mean = nn.Sequential(nn.Linear(self.ndf*2, self.ndf*2))              
        self.std = nn.Sequential(nn.Linear(self.ndf*2, self.ndf*2))    

    def forward(self, input):  
        batchSize = input.size()[0] 
        output = self.feature(input)

        mean_vector = self.mean(output.view(-1, 2*self.ndf))
        std_vector = self.std(output.view(-1, 2*self.ndf))
        if self.gpu>=0:
            mean_vector = mean_vector.cuda()
            std_vector = std_vector.cuda()

        return output.view(-1, 2*self.ndf), mean_vector, std_vector

"""
Classifier network
"""
class _netC(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netC, self).__init__()
        self.ndf = opt.ndf
        self.main = nn.Sequential(          
            nn.Linear(4*self.ndf, 2*self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.ndf, nclasses),                         
        )
        self.soft = nn.Sequential(nn.Sigmoid())

    def forward(self, input):      
        output_logit = self.main(input)
        output = self.soft(output_logit)
        return output_logit, output


"""
Feature extraction network for Sourceonly 
"""
class netF(nn.Module):
    def __init__(self, opt):
        super(netF, self).__init__()

        self.ndf = opt.ndf
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.ndf, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(self.ndf, self.ndf * 2, 5, 1, 0),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.feature(input)
        return output.view(-1, 2 * self.ndf)


"""
Classifier network for Sourceonly
"""
class netC(nn.Module):
    def __init__(self, opt, nclasses):
        super(netC, self).__init__()
        self.ndf = opt.ndf
        self.main = nn.Sequential(
            nn.Linear(2 * self.ndf, 2 * self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.ndf, nclasses),
        )

    def forward(self, input):
        output = self.main(input)
        return output