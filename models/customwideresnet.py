import torch
import torch.nn as nn
import torchvision.models as models

class CustomWideResNet(nn.Module):
    def __init__(self, model_name, widen_factor=2, num_classes=100):
        super(CustomWideResNet, self).__init__()
        if model_name == 'resnet18':
            original_resnet = models.resnet18(pretrained=False)
        elif model_name == 'resnet34':
            original_resnet = models.resnet34(pretrained=False)
        elif model_name == 'resnet50':
            original_resnet = models.resnet50(pretrained=False)
        elif model_name == 'resnet101':
            original_resnet = models.resnet101(pretrained=False)
        elif model_name == 'resnet152':
            original_resnet = models.resnet152(pretrained=False)

        self.conv1 = nn.Conv2d(3, int(64 * widen_factor), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        
        # Adjust the layers
        self.layer1 = self._make_layer(original_resnet.layer1, int(64 * widen_factor))
        self.layer2 = self._make_layer(original_resnet.layer2, int(128 * widen_factor))
        self.layer3 = self._make_layer(original_resnet.layer3, int(256 * widen_factor))
        self.layer4 = self._make_layer(original_resnet.layer4, int(512 * widen_factor))
        
        self.avgpool = original_resnet.avgpool
        self.fc = nn.Linear(int(512 * widen_factor * original_resnet.layer1.expansion), num_classes)
    
    def _make_layer(self, layer, out_channels):
        layers = []
        for block in layer:
            conv1 = nn.Conv2d(block.conv1.in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            bn1 = nn.BatchNorm2d(out_channels)
            conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            bn2 = nn.BatchNorm2d(out_channels)
            layers.append(nn.Sequential(conv1, bn1, self.relu, conv2, bn2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Example usage
# model = WideResNet18(width_multiplier=2)
# print(model)
def get_customwideresnet(model_name, widen_factor=2, num_classes=10):
    return CustomWideResNet(model_name, widen_factor, num_classes)
