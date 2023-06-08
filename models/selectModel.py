from models.cnn import get_model as get_cnn
from models.cnn_baseline import get_model as get_cnn_base
from models.MobileNetV3 import get_model as get_mobilenet
from models.cp_resnet import get_model as get_cp_resnet
from helpers.utils import NAME_TO_WIDTH

def get_model(config):
    if config.architecture == "cnn":
        return get_cnn(in_channels=config.in_channels,
                                n_classes=config.n_classes,
                                base_channels=config.base_channels,
                                channels_multiplier=config.channels_multiplier)
    elif config.architecture == "cnn_baseline":
        return get_cnn_base(in_channels=config.in_channels,
                                n_classes=config.n_classes,
                                base_channels=config.base_channels,
                                channels_multiplier=config.channels_multiplier)
    elif config.architecture == "mobilenet":
        pretrained_name = config.pretrained_name # e.g. mn10_as_mels_40
        if pretrained_name:
            return get_mobilenet(width_mult=NAME_TO_WIDTH(pretrained_name), pretrained_name=pretrained_name,
                                  head_type=config.head_type, se_dims=config.se_dims, num_classes=10)
        else:
            return get_mobilenet(width_mult=config.model_width, head_type=config.head_type, se_dims=config.se_dims,
                                  num_classes=config.n_classes)
    elif config.architecture == "cp_resnet":
        return get_cp_resnet(in_channels=config.in_channels,
                            n_classes=config.n_classes,
                            base_channels=config.base_channels,
                            channels_multiplier=config.channels_multiplier)
    return 0