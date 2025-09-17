import torch

def save_checkpoint (model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    #print (f"optim stae = {optimizer_state}")

    obj = (model_state, optimizer_state, iteration)
    torch.save (obj, out)


def load_checkpoint (src, model, optimizer ):
    obj = torch.load (src)
    model_state, optim_state, iteration = obj
    model.load_state_dict(model_state)
    optimizer.load_state_dict (optim_state)
    return iteration



if __name__ == '__main__':
    iteration=0
    model = torch.nn.Linear(3,3)
    optimizer = torch.optim.Adam (model.parameters())

    x = torch.randn(1,3)
    y = torch.randn(1,3)
    optimizer.zero_grad()
    loss = (model(x) - y).pow(2).sum()
    loss.backward()
    optimizer.step()  # <-- allocates state

    save_checkpoint(model, optimizer, iteration, f"checkpoints/itn_{iteration}.torch")

    model2 = torch.nn.Linear(3,3)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    iteration_loaded = load_checkpoint(f"checkpoints/itn_{iteration}.torch", model2, optimizer2)
    print (f"iteration_loaded back = {iteration_loaded}")

    assert model==model2
    assert optimizer==optimizer2