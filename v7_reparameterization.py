from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml

batch_size = 1
model = ""
yaml_file = ""
class_num = 1
save_model = ""
v7_tiny = True

# Select device
device = select_device('0', batch_size=batch_size)

# Load the trained model checkpoint
ckpt = torch.load(model, map_location=device)

# Initialize the model using the deploy configuration
model = Model(yaml_file, ch=3, nc=class_num).to(device)

# Load the deploy configuration YAML
with open('cfg/deploy/yolov7.yaml') as f:
    yml = yaml.load(f, Loader=yaml.SafeLoader)
anchors = len(yml['anchors'][0]) // 2

# Copy intersecting weights from the checkpoint to the model
state_dict = ckpt['model'].float().state_dict()
exclude = []
intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
model.load_state_dict(intersect_state_dict, strict=False)

# Update model attributes
model.names = ckpt['model'].names
model.nc = ckpt['model'].nc

# Reparameterize YOLO
if v7_tiny:
    for i in range((model.nc + 5) * anchors):
        model.state_dict()['model.77.m.0.weight'].data[i, :, :, :] *= state_dict['model.77.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.77.m.1.weight'].data[i, :, :, :] *= state_dict['model.77.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.77.m.2.weight'].data[i, :, :, :] *= state_dict['model.77.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.77.m.0.bias'].data += state_dict['model.77.m.0.weight'].mul(state_dict['model.77.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.77.m.1.bias'].data += state_dict['model.77.m.1.weight'].mul(state_dict['model.77.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.77.m.2.bias'].data += state_dict['model.77.m.2.weight'].mul(state_dict['model.77.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.77.m.0.bias'].data *= state_dict['model.77.im.0.implicit'].data.squeeze()
    model.state_dict()['model.77.m.1.bias'].data *= state_dict['model.77.im.1.implicit'].data.squeeze()
    model.state_dict()['model.77.m.2.bias'].data *= state_dict['model.77.im.2.implicit'].data.squeeze()
else:
    for i in range((model.nc + 5) * anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

# Prepare the checkpoint for saving
ckpt = {
    'model': deepcopy(model.module if is_parallel(model) else model).half(),
    'optimizer': None,
    'training_results': None,
    'epoch': -1
}

# Save the reparameterized model
torch.save(ckpt, save_model)
