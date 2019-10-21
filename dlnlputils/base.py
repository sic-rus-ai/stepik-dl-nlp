def get_params_number(model):
    return sum(t.numel() for t in model.parameters())
