from rnn_model import RNN
import torch
'''     Device configuration      '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 54
hidden_size = 64
num_layers=1
num_classes = 4

model=RNN(input_size,hidden_size,num_layers,num_classes).to(device)
model.load_state_dict(torch.load('./keytrack.pt'))
dummy_input = torch.randn(3, 32,54).to(device)
traced_script_module=torch.jit.trace(model,dummy_input)

traced_script_module.save('keytrack_trace.pt')