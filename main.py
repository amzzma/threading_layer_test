import threading
import torch
from torch import nn
from queue import Queue
from torch.utils.data import Subset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

q = Queue()

t = torch.tensor([0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
th_num = 3


def make_thread_weight(th_num, size):
    th_weights = []
    for i in range(th_num):
        th_weights.append(nn.Parameter(torch.randn((size, size))))
    return th_weights


thw = make_thread_weight(th_num, 28)
temp = torch.clone(thw[0])
# Here using .grad will raise warning because of accessing the non-leaf Tensor
temp_grad = torch.clone(thw[0].grad()) if thw[0].retain_grad() is not None else None


class threading_forward(threading.Thread):
    def __init__(self, x, i):
        super().__init__()
        self.weight = thw[i]
        self.x = x
        self.id = i
        # print("weight_grad:", self.weight.grad)

    def run(self):
        self.x = self.x.to(device)
        self.weight = self.weight.to(device)
        logit = self.x @ self.weight
        res = {"id": self.id, "logit": logit, "weight": self.weight}
        q.put(res)


class threading_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.threads = []
        self.th_num = th_num
        self.result = [0] * th_num

    def forward(self, x):
        for i in range(self.th_num):
            thread = threading_forward(x, i)
            self.threads.append(thread)
            thread.start()
        for thread in self.threads:
            thread.join()
        while q.qsize() < self.th_num:
            continue
        for i in range(self.th_num):
            temp = q.get()
            self.result[temp["id"]] = temp["logit"]
        self.result = torch.transpose(torch.stack(self.result), 0, 1)  # bs, th_num, c, hh, hw
        self.threads.clear()
        return self.result


class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.threading_block = threading_layer()
        self.dropout = nn.Dropout(p=0.5)
        self.conv = nn.Conv2d(th_num, 1, 1, 1, 0)
        self.ln1 = nn.Linear(28 * 28, 128)
        self.ln2 = nn.Linear(128, 64)
        self.ln3 = nn.Linear(64, 10)
        self.rl = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.threading_block(x))
        self.threading_block.result = [0] * th_num
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv(x)
        # print(x.shape)
        # x = x.view(-1, 3*28*28)
        x = torch.reshape(x, (-1, 28 * 28))
        x = self.rl(self.ln1(x))
        x = self.rl(self.ln2(x))
        x = self.ln3(x)
        return x


class CustomAdam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, closure=None):
        for i in range(th_num):
            thw[i].data = thw[i].data - thw[i].grad.data * 0.001
            thw[i].grad = None
            pass

        update = super().step()
        return update


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # transforms.Resize((128, 128))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

sub_trainset = Subset(trainset, range(6000))  # small sample
sub_trainloader = torch.utils.data.DataLoader(sub_trainset, batch_size=128, shuffle=True)

n = net()

print(n)
criterion = nn.CrossEntropyLoss()
optimizer = CustomAdam(n.parameters(), lr=0.001)

n.to(device)


def train(net, data: torch.utils.data.DataLoader, criterion, opt: torch.optim, epos, test):
    """
    :param test: If it is set True, you should comment 'thw[i].grad = None' in line 103.
    """
    global temp, temp_grad
    for epo in tqdm(range(epos), desc="Training", ncols=100):
        total_loss = []
        for i, j in enumerate(data):
            inputs, labels = j
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            if epo == 0 and i == 1:
                print("Does Gard in thw[0] is not None after opt.zero_grad(): ", torch.is_tensor(thw[0].grad)
                      and thw[0].grad.numel() > 0 and torch.all(thw[0].grad))
                print("--------> bottom <--------")
                if test == 1:
                    temp_grad = torch.clone(thw[0].grad) if thw[0].grad is not None else None

            o = net(inputs)
            loss = criterion(o, labels)
            total_loss.append(loss.item())
            loss.backward()

            if epo == 0 and i == 0:
                print("\n\n--------> test <--------")
                print("Does Gard in thw[0] is not None in init: ", torch.is_tensor(temp_grad)
                      and temp_grad > 0 and torch.all(temp_grad))

                print("Does Gard in thw[0] is not None after backward: ", torch.is_tensor(thw[0].grad)
                      and thw[0].grad.numel() > 0 and torch.all(thw[0].grad))

            if epo == 0 and i == 1 and test == 1:
                print("----> test by commenting 'thw[i].grad = None' in CustomAdam.step <----")
                print("Does thw[0] change after loss.backward() in batch 1: ", not torch.equal(temp, thw[0]))
                print("Does thw[0] Grad change after loss.backward() in batch 1: ", torch.equal(temp_grad, thw[0].grad))
                print("----> bottom <----")

            opt.step()
        total_loss = sum(total_loss) / len(total_loss)
        print(f"\nepoch{epo} loss:{total_loss}")


def calculate_metrics(model, test_loader):
    model.eval()
    correct = 0
    error = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images).to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            error += len(labels) - correct

    precision = correct / total if total > 0 else 0
    return precision


def show_model_static_dict(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":
    pass
    # print(len(trainset))
    show_model_static_dict(n)
    train(n, trainloader, criterion, optimizer, 20, 0)
    print(calculate_metrics(n, testloader))
