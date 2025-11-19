"""
Multi-scale CNN for capturing features at different resolutions
Inspired by Adegun et al. (2023) and Yang et al. (MSPPF-nets)
"""
import torch
import torch.nn as nn

class MultiScaleBranch(nn.Module):
    """Extract features at different scales"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MultiScaleBranch, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature fusion for remote sensing"""
    
    def __init__(self):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        self.branch1 = MultiScaleBranch(9, 32, kernel_size=3)
        self.branch2 = MultiScaleBranch(9, 32, kernel_size=5)
        self.branch3 = MultiScaleBranch(9, 32, kernel_size=7)
        
        self.fusion = nn.Conv2d(96, 128, kernel_size=1)
        self.bn_fusion = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)
        )
    
    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        fused = torch.cat([feat1, feat2, feat3], dim=1)
        fused = self.bn_fusion(self.fusion(fused))
        
        pooled = self.pool(fused)
        pooled = pooled.view(pooled.size(0), -1)
        
        output = self.classifier(pooled)
        return output
    
if __name__ == "__main__":
    # 1. Create an instance of your model
    model = MultiScaleFeatureExtractor()
    print("Model instance created successfully.")
    
    # 2. Create some dummy input data to test with
    #    The model expects 9 input channels (from your branch1, 2, 3)
    #    Let's create a "batch" of 2 images, with 9 channels, and 64x64 pixels
    dummy_input = torch.randn(2, 9, 64, 64) 
    print(f"Created dummy input tensor with shape: {dummy_input.shape}")

    # 3. Pass the dummy data through the model
    output = model(dummy_input)
    
    # 4. Print the results
    print(f"Model output shape: {output.shape}")
    print("Model output tensor (first item in batch):")
    print(output[0])
