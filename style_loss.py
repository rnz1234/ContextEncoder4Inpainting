import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):
    bs, fm, fmd1, fmd2 = input.size()  
    # bs : batch size (=1)
    # fm : number of feature maps
    # fmd1,fmd2 : dimensions of a f. map (N=fmd1*fmd2)

    # features = input.view(bs * fm, fmd1 * fmd2)  # resize F_XL into \hat F_XL
    # gram_m = torch.mm(features, features.t())  # compute the gram product

    # features = input.view(bs, fm, fmd1 * fmd2)  # resize F_XL into \hat F_XL
    # gram_m = torch.stack([torch.mm(features[i], features[i].t()) for i in range(bs)])  # compute the gram product

    # return gram_m.div(fm * fmd1 * fmd2) #  bs * fm * fmd1 * fmd2

    features = input.view(bs, fm, fmd1 * fmd2)  # resize F_XL into \hat F_XL
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (fm * fmd1 * fmd2)
    return gram

    # normalize the values of the gram matrix, by dividing by the number of element in each feature maps.
   


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input, target):
        # import pdb 
        # pdb.set_trace()
        input_g = gram_matrix(input)
        target_g = gram_matrix(target).detach()
        return F.mse_loss(input_g, target_g)