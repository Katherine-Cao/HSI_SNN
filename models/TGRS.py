
"""
@CreatedDate:   2022/04
@Author: Katherine_Cao(https://github.com/Katherine-Cao/HSI_SNN)
"""
import torch
import torch.nn as nn

class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(1 - (input * input)) < 0.7
        return grad_input * temp.float()

def channel_shuffle(x, groups: int):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x


class TGRS(nn.Module):
    def __init__(self, num_steps, leak_mem, img_size, num_cls, input_dim):
        super(TGRS, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem

        # (">>>>>>>>>>>>>>>>>>> SNN Direct Coding For TGRS >>>>>>>>>>>>>>>>>>>>>>")

        bias_flag = False

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)

        # (">>>>>>>>>>>>>>>>>>> branch1_left >>>>>>>>>>>>>>>>>>>>>>")
        # self.branch1 = nn.Sequential(
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        # )

        # (">>>>>>>>>>>>>>>>>>> branch1_right >>>>>>>>>>>>>>>>>>>>>>")
        # self.branch2 = nn.Sequential(
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False, groups=32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=False)
        # )

        self.conv6 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        # (">>>>>>>>>>>>>>>>>>> branch2_left >>>>>>>>>>>>>>>>>>>>>>")
        # self.branch3 = nn.Sequential(
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        # )

        # (">>>>>>>>>>>>>>>>>>> branch2_right >>>>>>>>>>>>>>>>>>>>>>")
        # self.branch4 = nn.Sequential(
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, groups=64)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # )

        self.conv11 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(256 * 3 * 3, self.num_cls, bias=False)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, self.conv11, ]

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)
            elif isinstance(m, nn.Linear):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=5)


    def forward(self, input):
        batch_size = input.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv4 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv5 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv6 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv7 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv8 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv9 = torch.zeros(batch_size, 64, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv10 = torch.zeros(batch_size, 64, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv11 = torch.zeros(batch_size, 256, self.img_size // 2, self.img_size // 2).cuda()
        mem_fc1 = torch.zeros(batch_size, self.num_cls).cuda()


        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8, mem_conv9, mem_conv10, mem_conv11]

        static_input1 = self.conv1(input)

        for t in range(self.num_steps):
            mem_conv_list[0] = self.leak_mem * mem_conv_list[0] + (1 - self.leak_mem) * static_input1                   # 总分支
            mem_thr = mem_conv_list[0] - self.conv_list[0].threshold
            out = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[0]).cuda()
            rst[mem_thr > 0] = self.conv_list[0].threshold
            mem_conv_list[0] = mem_conv_list[0] - rst
            out_prev = out.clone()

            x1, x2 = out_prev.chunk(2, dim=1)                                                                           # x1 左分支  x2 右分支

            mem_conv_list[1] = self.leak_mem * mem_conv_list[1] + (1 - self.leak_mem) * self.conv2(x1)                  # 左分支1.1
            mem_thr = mem_conv_list[1] - self.conv_list[1].threshold
            x1 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[1]).cuda()
            rst[mem_thr > 0] = self.conv_list[1].threshold
            mem_conv_list[1] = mem_conv_list[1] - rst
            x1_prev = x1.clone()

            mem_conv_list[2] = self.leak_mem * mem_conv_list[2] + (1 - self.leak_mem) * self.conv3(x1_prev)             # 左分支1.2
            mem_thr = mem_conv_list[2] - self.conv_list[2].threshold
            out1 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[2]).cuda()
            rst[mem_thr > 0] = self.conv_list[2].threshold
            mem_conv_list[2] = mem_conv_list[2] - rst
            out1_prev = out1.clone()

            mem_conv_list[3] = self.leak_mem * mem_conv_list[3] + (1 - self.leak_mem) * self.conv4(x2)                  # 右分支1.1
            mem_thr = mem_conv_list[3] - self.conv_list[3].threshold
            x2 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[3]).cuda()
            rst[mem_thr > 0] = self.conv_list[3].threshold
            mem_conv_list[3] = mem_conv_list[3] - rst
            x2_prev = x2.clone()

            mem_conv_list[4] = self.leak_mem * mem_conv_list[4] + (1 - self.leak_mem) * self.conv5(x2_prev)             # 右分支1.2
            mem_thr = mem_conv_list[4] - self.conv_list[4].threshold
            out2 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[4]).cuda()
            rst[mem_thr > 0] = self.conv_list[4].threshold
            mem_conv_list[4] = mem_conv_list[4] - rst
            out2_prev = out2.clone()

            out_prev = torch.cat((out1_prev, out2_prev), 1)                                                             # 分支汇总
            out = channel_shuffle(out_prev, 2)                                                                          # 通道打乱

            mem_conv_list[5] = self.leak_mem * mem_conv_list[5] + (1 - self.leak_mem) * self.conv6(out)                 # 总分支
            mem_thr = mem_conv_list[5] - self.conv_list[5].threshold
            out = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[5]).cuda()
            rst[mem_thr > 0] = self.conv_list[5].threshold
            mem_conv_list[5] = mem_conv_list[5] - rst
            out_prev = out.clone()

            out = self.pool1(out_prev)                                                                                  # 池化层1
            out = out.clone()

            x3, x4 = out.chunk(2, dim=1)                                                                                # x3 左分支  x4 右分支

            mem_conv_list[6] = self.leak_mem * mem_conv_list[6] + (1 - self.leak_mem) * self.conv7(x3)                  # 左分支2.1
            mem_thr = mem_conv_list[6] - self.conv_list[6].threshold
            x3 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[6]).cuda()
            rst[mem_thr > 0] = self.conv_list[6].threshold
            mem_conv_list[6] = mem_conv_list[6] - rst
            x3_prev = x3.clone()

            mem_conv_list[7] = self.leak_mem * mem_conv_list[7] + (1 - self.leak_mem) * self.conv8(x3_prev)             # 左分支2.2
            mem_thr = mem_conv_list[7] - self.conv_list[7].threshold
            out1 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[7]).cuda()
            rst[mem_thr > 0] = self.conv_list[7].threshold
            mem_conv_list[7] = mem_conv_list[7] - rst
            out1_prev = out1.clone()

            mem_conv_list[8] = self.leak_mem * mem_conv_list[8] + (1 - self.leak_mem) * self.conv9(x4)                  # 右分支2.1
            mem_thr = mem_conv_list[8] - self.conv_list[8].threshold
            x4 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[8]).cuda()
            rst[mem_thr > 0] = self.conv_list[8].threshold
            mem_conv_list[8] = mem_conv_list[8] - rst
            x4_prev = x4.clone()

            mem_conv_list[9] = self.leak_mem * mem_conv_list[9] + (1 - self.leak_mem) * self.conv10(x4_prev)            # 左分支2.2
            mem_thr = mem_conv_list[9] - self.conv_list[9].threshold
            out2 = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[9]).cuda()
            rst[mem_thr > 0] = self.conv_list[9].threshold
            mem_conv_list[9] = mem_conv_list[9] - rst
            out2_prev = out2.clone()

            out_prev = torch.cat((out1_prev, out2_prev), 1)  # 分支汇总
            out = channel_shuffle(out_prev, 2)

            mem_conv_list[10] = self.leak_mem * mem_conv_list[10] + (1 - self.leak_mem) * self.conv11(out)              # 总分支
            mem_thr = mem_conv_list[10] - self.conv_list[10].threshold
            out = self.spike_fn(mem_thr)
            # Soft reset
            rst = torch.zeros_like(mem_conv_list[10]).cuda()
            rst[mem_thr > 0] = self.conv_list[10].threshold
            mem_conv_list[10] = mem_conv_list[10] - rst
            out_prev = out.clone()

            out = self.pool1(out_prev)                                                                                  # 池化层1
            out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = mem_fc1 + self.fc1(out_prev)

        out_voltage = mem_fc1 / self.num_steps

        return out_voltage
