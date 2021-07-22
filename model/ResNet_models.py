import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet, UNetConvBlock, UNetUpBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from utils import gradient_x, gradient_y, normalise




class FCDiscriminator(nn.Module):
    def __init__(self, ndf = 16):
        super(FCDiscriminator, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # self.upsample = F.interpolate(scale_factor=2, mode='bilinear', align_corners=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.down_block_1 = UNetConvBlock(1, ndf)
        self.down_block_2 = UNetConvBlock(ndf, 2*ndf)
        self.down_block_3 = UNetConvBlock(2*ndf, 4*ndf)
        self.down_block_4 = UNetConvBlock(4*ndf, 8*ndf)
        self.down_block_5 = UNetConvBlock(8*ndf, 16*ndf)

        self.up_block_4 = UNetUpBlock(16*ndf, 8*ndf)
        self.up_block_3 = UNetUpBlock(8*ndf, 4*ndf)
        self.up_block_2 = UNetUpBlock(4*ndf, 2*ndf)
        self.up_block_1 = UNetUpBlock(2*ndf, ndf)

        self.conv1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4*ndf, 4*ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=2, padding=1)


        # self.conv1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(4*ndf, 4*ndf, kernel_size=3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=2, padding=1)

        # self.final_layer = Classifier_Module([6, 12, 18, 24], [6, 12, 18, 24], 1, ndf)
        self.final_layer = nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)

        # self.dropout_1_1 = nn.Dropout(p=0.3)
        # self.dropout_1_2 = nn.Dropout(p=0.3)
        # self.dropout_1_3 = nn.Dropout(p=0.3)
        # self.dropout_1_4 = nn.Dropout(p=0.3)
        # self.dropout_1_5 = nn.Dropout(p=0.3)
        # self.dropout_2_4 = nn.Dropout(p=0.3)
        # self.dropout_2_3 = nn.Dropout(p=0.3)
        # self.dropout_2_2 = nn.Dropout(p=0.3)
        # self.dropout_2_1 = nn.Dropout(p=0.3)

    def forward(self, x):
        block_1 = self.down_block_1(x) # ndf*256*256
        x = self.conv1(block_1)
        # print("block_1: ", block_1.size(), "   After maxpool: ", x.size())

        block_2 = self.down_block_2(x)
        x = self.conv2(block_2)
        # print("block_2: ", block_2.size(), "   After maxpool: ", x.size())

        block_3 = self.down_block_3(x)
        x = self.conv3(block_3)
        # print("block_3: ", block_3.size(), "   After maxpool: ", x.size())

        block_4 = self.down_block_4(x)
        x = self.conv4(block_4)
        # print("block_4: ", block_4.size(), "   After maxpool: ", x.size())

        block_5 = self.down_block_5(x)
        # print("block_5: ", block_5.size())

        out = self.up_block_4(block_5, block_4)
        out = self.up_block_3(out, block_3)
        out = self.up_block_2(out, block_2)
        out = self.up_block_1(out, block_1)

        out = self.final_layer(out)

        return out


    #     super(FCDiscriminator, self).__init__()
    #     self.activation = F.relu
    #
    #     # self.pool1 = nn.MaxPool2d(2)
    #     # self.pool2 = nn.MaxPool2d(2)
    #     # self.pool3 = nn.MaxPool2d(2)
    #     # self.pool4 = nn.MaxPool2d(2)
    #     self.down1 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
    #     self.down2 = nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=2, padding=1)
    #     self.down3 = nn.Conv2d(4*ndf, 4*ndf, kernel_size=3, stride=2, padding=1)
    #     self.down4 = nn.Conv2d(8*ndf, 8*ndf, kernel_size=3, stride=2, padding=1)
    #
    #     self.conv_block1_64 = UNetConvBlock(1, ndf)
    #     self.conv_block64_128 = UNetConvBlock(ndf, 2*ndf)
    #     self.conv_block128_256 = UNetConvBlock(2*ndf, 4*ndf)
    #     self.conv_block256_512 = UNetConvBlock(4*ndf, 8*ndf)
    #     self.conv_block512_1024 = UNetConvBlock(8*ndf, 16*ndf)
    #     # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
    #     self.up_block1024_512 = UNetUpBlock(16*ndf, 8*ndf)
    #     self.up_block512_256 = UNetUpBlock(8*ndf, 4*ndf)
    #     self.up_block256_128 = UNetUpBlock(4*ndf, 2*ndf)
    #     self.up_block128_64 = UNetUpBlock(2*ndf, ndf)
    #
    #     self.last = nn.Conv2d(ndf, 1, 1, stride=1)
    #
    # def forward(self, x):
    #     #         print 'line 70 ',x.size()
    #     block1 = self.conv_block1_64(x)
    #     # print("block1: ", block1.size())
    #     pool1 = self.down1(block1)
    #
    #     block2 = self.conv_block64_128(pool1)
    #     # print("block2: ", block2.size())
    #     pool2 = self.down2(block2)
    #
    #     block3 = self.conv_block128_256(pool2)
    #     # print("block3: ", block3.size())
    #     pool3 = self.down3(block3)
    #
    #     block4 = self.conv_block256_512(pool3)
    #     # print("block4: ", block4.size())
    #     pool4 = self.down4(block4)
    #
    #     block5 = self.conv_block512_1024(pool4)
    #
    #     up = self.up_block1024_512(block5, block4)
    #     # print("up1: ", up.size())
    #
    #     up = self.up_block512_256(up, block3)
    #     # print("up2: ", up.size())
    #
    #     up = self.up_block256_128(up, block2)
    #     # print("up3: ", up.size())
    #
    #     up = self.up_block128_64(up, block1)
    #     # print("up4: ", up.size())
    #
    #     return self.last(up)


    # def __init__(self, ndf = 64):
    #     super(FCDiscriminator, self).__init__()
    #     self.conv1 = nn.Conv2d(1, ndf, kernel_size=3, stride=2, padding=1)
    #     self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
    #     self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
    #     self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
    #     self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    #     self.bn1 = nn.BatchNorm2d(ndf)
    #     self.bn2 = nn.BatchNorm2d(ndf)
    #     self.bn3 = nn.BatchNorm2d(ndf)
    #     self.bn4 = nn.BatchNorm2d(ndf)
    #     #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
    #     # #self.sigmoid = nn.Sigmoid()
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.leaky_relu(x)
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.leaky_relu(x)
    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = self.leaky_relu(x)
    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = self.leaky_relu(x)
    #     x = self.classifier(x)
    #     return x




class Descriptor(nn.Module):
    def __init__(self, channel):
        super(Descriptor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sconv1 = nn.Conv2d(3, channel, kernel_size=3, stride=2, padding=1)
        self.sconv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sconv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.sconv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.bn4 = nn.BatchNorm2d(1024)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 1024)
        self.conv_pred = nn.Conv2d(1, channel, 3, 1,1)
        self.conv1 = nn.Conv2d(channel*2, channel, 3, 2, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, channel , 3, 2, 1)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, 1, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, input, seg):
        x1 = self.sconv1(input)
        # # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        # x2 = self.sconv2(x1)
        # # x2 = self.bn2(x2)
        # x2 = self.relu(x2)
        # x2 = self.maxpool(x2)
        # x3 = self.sconv3(x2)
        # # x3 = self.bn3(x3)
        # x3 = self.relu(x3)
        # x3 = self.maxpool(x3)
        # x4 = self.sconv4(x3)
        # # x4 = self.bn4(x4)
        # x4 = self.relu(x4)
        # x5 = self.layer5(x4)
        int_feat = self.upsample(x1)
        seg_conv = self.conv_pred(seg)
        feature_map = torch.cat((int_feat,seg_conv),1)
        x = self.conv1(feature_map)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        return x

class Generator(nn.Module):
    def __init__(self, channel):
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder_v1(channel)

    def forward(self, x):
        self.sal_init, self.sal_ref = self.sal_encoder(x)
        self.sal_init = F.upsample(self.sal_init, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        self.sal_ref = F.upsample(self.sal_ref, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        # self.bd_pred = F.upsample(self.bd_pred, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        # self.sal_grad = F.upsample(self.sal_grad, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return self.sal_init, self.sal_ref

class CAM_Module(nn.Module):
    """ Channel attention module"""
    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self,in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            # out += self.conv2d_list[i+1](x)
            out = out + self.conv2d_list[i + 1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        # res += x
        res = res + x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*3)

        # self.conv1 = nn.Conv2d(256, channel, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # self.conv1 = Triple_Conv(256, channel)
        # self.conv2 = Triple_Conv(512, channel)
        # self.conv3 = Triple_Conv(1024, channel)
        # self.conv4 = Triple_Conv(2048, channel)


        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pam_attention5 = PAM_Module(channel)
        self.pam_attention4 = PAM_Module(channel)
        self.pam_attention3 = PAM_Module(channel)
        self.pam_attention2 = PAM_Module(channel)

        self.cam_attention4 = CAM_Module(channel)
        self.cam_attention3 = CAM_Module(channel)
        self.cam_attention2 = CAM_Module(channel)


        self.pam_attention1 = PAM_Module(channel)
        self.racb_layer = RCAB(channel * 4)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = Triple_Conv(2*channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()
        self.conv4_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.pam_attention4_2 = PAM_Module(channel)
        self.pam_attention3_2 = PAM_Module(channel)
        self.pam_attention2_2 = PAM_Module(channel)
        self.cam_attention4_2 = CAM_Module(channel)
        self.cam_attention3_2 = CAM_Module(channel)
        self.cam_attention2_2 = CAM_Module(channel)
        self.racb_43_2 = RCAB(channel * 2)
        self.racb_432_2 = RCAB(channel * 3)
        self.racb_4321_2 = RCAB(channel * 4)
        self.conv43_2 = Triple_Conv(2 * channel, channel)
        self.conv432_2 = Triple_Conv(3 * channel, channel)
        self.conv4321_2 = Triple_Conv(4 * channel, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        # Boundary Prediction
        self.bd_pred = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)
        self.bd_fuse = Triple_Conv(2, 1)


        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        conv1_feat = self.conv1(x1)  # 32 * 64 * 64
        conv2_feat = self.conv2(x2)
        conv2_feat1 = self.pam_attention2(conv2_feat)
        conv2_feat2 = self.cam_attention2(conv2_feat)
        conv2_feat = conv2_feat1+conv2_feat2  # 32 * 32 * 32
        conv3_feat = self.conv3(x3)
        conv3_feat1 = self.pam_attention3(conv3_feat)
        conv3_feat2 = self.cam_attention3(conv3_feat)
        conv3_feat = conv3_feat1+conv3_feat2  # 32 * 16 * 16
        conv4_feat = self.conv4(x4)
        conv4_feat1 = self.pam_attention4(conv4_feat)
        conv4_feat2 = self.cam_attention4(conv4_feat)
        conv4_feat = conv4_feat1+conv4_feat2
        conv4_feat = self.upsample2(conv4_feat)  # 32 * 16 * 16

        conv43 = torch.cat((conv4_feat, conv3_feat),1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)  # 32 * 32 * 32
        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)  # 96 * 32 * 32

        # conv432 = self.conv432(conv432)



        # # Boundary Prediction
        # conv432_bd = self.conv432(conv432)
        # conv432_bd = self.upsample2(conv432_bd)
        # conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432_bd, conv1_feat), 1)
        #
        # conv4321 = self.racb_4321(conv4321)
        # # conv4321 = self.conv4321(conv4321)
        # bd_pred = self.bd_pred(conv4321)



        # conv432 = self.upsample2(conv432)
        sal_init = self.layer6(conv432)

        x2_2 = self.HA(sal_init.sigmoid(), x2)
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 8 x 8

        conv2_feat = self.conv2_2(x2_2)
        conv2_feat1 = self.pam_attention2_2(conv2_feat)
        conv2_feat2 = self.cam_attention2_2(conv2_feat)
        conv2_feat = conv2_feat1+conv2_feat2
        conv3_feat = self.conv3_2(x3_2)
        conv3_feat1 = self.pam_attention3_2(conv3_feat)
        conv3_feat2 = self.cam_attention3_2(conv3_feat)
        conv3_feat = conv3_feat1+conv3_feat2
        conv4_feat = self.conv4_2(x4_2)
        conv4_feat1 = self.pam_attention4_2(conv4_feat)
        conv4_feat2 = self.cam_attention4_2(conv4_feat)
        conv4_feat = conv4_feat1+conv4_feat2

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43_2(conv43)
        conv43 = self.conv43_2(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432_2(conv432)

        conv432 = self.conv432_2(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321_2(conv4321)
        sal_ref = self.layer7(conv4321)


        # Compute the gradient in the sal_ref map

        # sal_grad_x = gradient_x(sal_ref)
        # sal_grad_y = gradient_y(sal_ref)
        # sal_grad = torch.abs(sal_grad_x) + torch.abs(sal_grad_y)
        # sal_grad = normalise(sal_grad)


        # Fuse the gradient in sal_ref and boundary prediction
        # bd_pred = self.bd_fuse((torch.cat((sal_grad, bd_pred), 1)))


        return self.upsample8(sal_init), self.upsample4(sal_ref)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)






class Saliency_feat_encoder_v1(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(Saliency_feat_encoder_v1, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*3)


        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)


        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.pam_attention5 = PAM_Module(channel)
        # self.pam_attention4 = PAM_Module(channel)
        # self.pam_attention3 = PAM_Module(channel)
        # self.pam_attention2 = PAM_Module(channel)
        #
        # self.cam_attention4 = CAM_Module(channel)
        # self.cam_attention3 = CAM_Module(channel)
        # self.cam_attention2 = CAM_Module(channel)

        # self.pam_attention1 = PAM_Module(channel)
        self.racb_layer = RCAB(channel * 4)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = Triple_Conv(2*channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        # self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()
        self.conv4_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        # self.pam_attention4_2 = PAM_Module(channel)
        # self.pam_attention3_2 = PAM_Module(channel)
        # self.pam_attention2_2 = PAM_Module(channel)
        # self.cam_attention4_2 = CAM_Module(channel)
        # self.cam_attention3_2 = CAM_Module(channel)
        # self.cam_attention2_2 = CAM_Module(channel)
        self.racb_43_2 = RCAB(channel * 2)
        self.racb_432_2 = RCAB(channel * 3)
        self.racb_4321_2 = RCAB(channel * 4)
        self.conv43_2 = Triple_Conv(2 * channel, channel)
        self.conv432_2 = Triple_Conv(3 * channel, channel)
        self.conv4321_2 = Triple_Conv(4 * channel, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        # Boundary Prediction
        # self.bd_pred = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)
        # self.bd_fuse = Triple_Conv(2, 1)


        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        conv1_feat = self.conv1(x1)  # 32 * 64 * 64
        conv2_feat = self.conv2(x2)
        # conv2_feat1 = self.pam_attention2(conv2_feat)
        # conv2_feat2 = self.cam_attention2(conv2_feat)
        # conv2_feat = conv2_feat1+conv2_feat2  # 32 * 32 * 32
        conv3_feat = self.conv3(x3)
        # conv3_feat1 = self.pam_attention3(conv3_feat)
        # conv3_feat2 = self.cam_attention3(conv3_feat)
        # conv3_feat = conv3_feat1+conv3_feat2  # 32 * 16 * 16
        conv4_feat = self.conv4(x4)
        # conv4_feat1 = self.pam_attention4(conv4_feat)
        # conv4_feat2 = self.cam_attention4(conv4_feat)
        # conv4_feat = conv4_feat1+conv4_feat2
        conv4_feat = self.upsample2(conv4_feat)  # 32 * 16 * 16

        conv43 = torch.cat((conv4_feat, conv3_feat),1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)  # 32 * 32 * 32
        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)  # 96 * 32 * 32

        sal_init = self.layer6(conv432)

        x2_2 = self.HA(sal_init.sigmoid(), x2)
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 8 x 8

        conv2_feat = self.conv2_2(x2_2)
        # conv2_feat1 = self.pam_attention2_2(conv2_feat)
        # conv2_feat2 = self.cam_attention2_2(conv2_feat)
        # conv2_feat = conv2_feat1+conv2_feat2
        conv3_feat = self.conv3_2(x3_2)
        # conv3_feat1 = self.pam_attention3_2(conv3_feat)
        # conv3_feat2 = self.cam_attention3_2(conv3_feat)
        # conv3_feat = conv3_feat1+conv3_feat2
        conv4_feat = self.conv4_2(x4_2)
        # conv4_feat1 = self.pam_attention4_2(conv4_feat)
        # conv4_feat2 = self.cam_attention4_2(conv4_feat)
        # conv4_feat = conv4_feat1+conv4_feat2

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43_2(conv43)
        conv43 = self.conv43_2(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432_2(conv432)

        conv432 = self.conv432_2(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321_2(conv4321)
        sal_ref = self.layer7(conv4321)

        return self.upsample8(sal_init), self.upsample4(sal_ref)


    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
