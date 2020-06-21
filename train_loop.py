from train_functions import *


# loss choice
loss_type = 'PAT'
train, is_adv = which_train(loss_type)
# net initialization
# Generator
batch_size = 16    # Keep in mind: patches 32x32 already preprocessed
net = FCNN(input_channels=3, batch_size=batch_size)
net.float()
if torch.cuda.is_available():
    net.cuda()
net.apply(init_weights)
net.train()
# Discriminator
if is_adv:
    disc = Discriminator()
    disc.float()
    disc.cuda()
    optim_d = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
# data loader
data = COCO('data/train/', 'data/target/')
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
# parameters setting
epochs = 1
lr = 1e-4
betas = (0.5, 0.999)
optim_g = optim.Adam(net.parameters(), lr=lr, betas=betas)
load_weights = False
state_dict = 'state_1e_E'
# technical details
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if load_weights:
    print('Loading {x}'.format(x=state_dict))
    net.load_state_dict(torch.load('trained_models/{x}.pth'.format(x=state_dict), map_location=torch.device('cpu')))
    starting_epoch = int(re.sub("[^0-9]", "", state_dict))
# training procedure
try:
    for e in range(epochs):
        start = time.perf_counter()
        start_step = start
        print('Epoch %d' % (e + 1))
        if is_adv:
            train(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch=e, epochs=epochs,
                  batch_size=batch_size)
        else:
            train(net, None, optim_g, None, device, data_loader, start_step, current_epoch=e, epochs=epochs,
                  batch_size=batch_size)
        end = time.perf_counter()
        print('Epoch %d ended, elapsed time: %f seconds.' % (e + 1, round((end - start), 2)))

except KeyboardInterrupt:
    print('Training interrupted. Saving model.')
    today = date.today()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    torch.save(net.state_dict(), 'state_interrupt_{mode}_{date}_{time}.pth'.format(mode=''.join(loss_type),
                                                                                   date=today.strftime("%b-%d-%Y"),
                                                                                   time=current_time))
print('Saving checkpoint.')
today = date.today()
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
torch.save(net.state_dict(), 'state_{d}e_{mode}_{date}_{time}.pth'.format(d=epochs,
                                                                          mode=''.join(loss_type),
                                                                          date=today.strftime("%b-%d-%Y"),
                                                                          time=current_time))
