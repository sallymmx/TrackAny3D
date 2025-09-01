
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, input_dim, reduction_factor=2):
        super(Adapter, self).__init__()
        # 降维后的维度
        reduced_dim = input_dim // reduction_factor

        # FC Down: 降维全连接层
        self.fc_down = nn.Linear(input_dim, reduced_dim)

        # 激活函数
        self.activation = nn.GELU()

        # FC Up: 升维全连接层
        self.fc_up = nn.Linear(reduced_dim, input_dim)
        nn.init.xavier_normal_(self.fc_down.weight)
        nn.init.xavier_normal_(self.fc_up.weight)
        if self.fc_down.bias is not None:
            nn.init.zeros_(self.fc_down.bias)  # 初始化 fc_down 的偏置为 0
        if self.fc_up.bias is not None:
            nn.init.zeros_(self.fc_up.bias)  # 初始化 fc_up 的偏置为 0
    def forward(self, x):
        # 保存输入用于残差连接
        x = x.permute(0, 2, 1)
        # 前向传播

        out = self.fc_down(x)
        out = self.activation(out)
        out = self.fc_up(out)

        # 残差连接
        out += x
        out =out.permute(0, 2, 1)
        return out
class Up(nn.Module):
    def __init__(self, input_channel, per_point_mlp1,per_point_mlp2):
        super(Up, self).__init__()
        # 降维后的维度
        self.seq_per_point = nn.ModuleList()
        in_channel = input_channel
        for out_channel in per_point_mlp1:  # per_point_mlp1=[64, 64, 64, 128, 1024]
            self.seq_per_point.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        in_channel = in_channel + per_point_mlp1[1]

        self.seq_per_point2 = nn.ModuleList()
        for out_channel in per_point_mlp2:
            self.seq_per_point2.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel


        self.fc = nn.Conv1d(in_channel, per_point_mlp2[-1], 1)

    def forward(self, x):
        second_layer_out = None
        for i, mlp in enumerate(self.seq_per_point):
            x = mlp(x)
            if i == 1:
                second_layer_out = x
        pooled_feature = self.pool(x)  # B,C,1
        pooled_feature_expand = pooled_feature.expand_as(x)
        x = torch.cat([second_layer_out, pooled_feature_expand], dim=1)
        for mlp in self.seq_per_point2:
            x = mlp(x)

        x = self.fc(x)

        return x
class Down(nn.Module):
    def __init__(self, input_channel, per_point_mlp1, per_point_mlp2):
        super(Down, self).__init__()
        # 降维后的维度
        self.seq_per_point = nn.ModuleList()
        in_channel = input_channel
        for out_channel in per_point_mlp1:  # per_point_mlp1=[64, 64, 64, 128, 1024]
            self.seq_per_point.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        in_channel = in_channel + per_point_mlp1[1]

        self.seq_per_point2 = nn.ModuleList()
        for out_channel in per_point_mlp2:
            self.seq_per_point2.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU()
                ))
            in_channel = out_channel

        self.fc = nn.Conv1d(in_channel, per_point_mlp2[-1], 1)
    def forward(self, x):
        second_layer_out = None
        for i, mlp in enumerate(self.seq_per_point):
            x = mlp(x)
            if i == 1:
                second_layer_out = x
        pooled_feature = self.pool(x)  # B,C,1
        pooled_feature_expand = pooled_feature.expand_as(x)
        x = torch.cat([second_layer_out, pooled_feature_expand], dim=1)
        for mlp in self.seq_per_point2:
            x = mlp(x)

        x = self.fc(x)

        return x
if __name__ == '__main__':
# 测试代码
    input_tensor = torch.randn(32, 128)  # 假设输入为32个样本，每个样本有128维特征
    adapter = Adapter(input_dim=128)
    output = adapter(input_tensor)
    print(output.shape)  # 输出应该是 (32, 128)
