import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models import *
import math
import gc



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(512, 50)
        self.act = nn.ReLU()
        self.lr2 = nn.Linear(50, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr1(x)
        x = self.act(x)
        x = self.lr2(x)
        x = self.sm(x)
        return x

z_dim = 512
w_dim = 512
mapping_layers = 8
mapping_activation = "LeakyReLU"
device = "cuda" if torch.cuda.is_available() else "cpu"


def save_tensor_images(image_tensor, size, num_images, nrow, time, detail):

    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow, padding=2, pad_value=1, scale_each=True)

    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")

    fig = plt.gcf()
    fig.set_size_inches(image_grid.shape[2] / 100, image_grid.shape[1] / 100)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig("./plots/save"+str(num_images)+str(time)+str(detail)+".jpg", dpi=100, pad_inches=0)
    plt.close()
    del image_grid
    del image_unflat
    gc.collect()


def generate(num, nrow, time, detail):
    gen = Generator_without_noise_mapping().to(device)
    stage = 6
    gen.load_state_dict(torch.load("GenNet.pkl"))
    gen.eval()

    map = MappingNet().to(device)
    map.load_state_dict(torch.load("MappingNet.pkl"))
    map.eval()

    mlp_all = MLP()
    mlp_all.load_state_dict(torch.load("mlp_for_ALL.pkl"))
    mlp_all.eval()

    mlp_night = MLP()
    mlp_night.load_state_dict(torch.load("mlp_for_Night.pkl"))
    mlp_night.eval()

    mlp_day = MLP()
    mlp_day.load_state_dict(torch.load("mlp_for_Day.pkl"))
    mlp_day.eval()

    z = np.random.randn(num, z_dim)
    z = torch.from_numpy(z).float().to(device)
    with torch.no_grad():
        w = map.forward(z)
        if time != 4:
            mask = check_time(w[:, 0, :].cpu().numpy(), time, detail, mlp_all, mlp_night, mlp_day)
            w = w[mask, :, :]
            while w.shape[0] < num:
                z = np.random.randn(num - w.shape[0], z_dim)
                z = torch.from_numpy(z).float().to(device)
                w_new = map.forward(z)
                mask = check_time(w_new[:, 0, :].cpu().numpy(), time, detail, mlp_all, mlp_night, mlp_day)
                w_new = w_new[mask, :, :]
                w = torch.cat([w, w_new], dim=0)
        images = gen.forward(w[:4, :, :], stage, 1, truncation_psi=0.8, noise=1)
        for i in range(1, math.ceil(num/4)): # just to save memory
            image = gen.forward(w[i * 4: i * 4 + 4, :, :], stage, 1, truncation_psi=0.8, noise=1)
            images = torch.cat([images, image], dim=0)
        save_tensor_images(images, (3, images.shape[2], images.shape[2]), num, nrow, time, detail)
    del gen
    del map
    del mlp_all
    del mlp_day
    del mlp_night
    del images
    del w
    gc.collect()


def check_time(w, time, detail, mlp_all, mlp_night, mlp_day):


    w = torch.tensor(w).float()
    res = mlp_all(w)
    res = np.where(res > 0.5, 1, 0).reshape(-1)
    res = res.astype(np.int)
    res = res == np.full_like(res, time)
    if detail != 4:
        if time == 0:
            r = mlp_night(w)
            r = np.where(r > 0.5, 1, 0).reshape(-1)
            res = np.logical_and(res, r.astype(np.int) == np.full_like(res, detail))
        if time == 1:
            r = mlp_day(w)
            r = np.where(r > 0.5, 1, 0).reshape(-1)
            res = np.logical_and(res, r.astype(np.int) == np.full_like(res, detail))


    return res



