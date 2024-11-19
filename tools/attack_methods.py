import algorithm.attack as attack


class NoneMethod():
    def __init__(self, model=None):
        super().__init__()
        self.name = 'none'
        self.model = model

    def untarget(self, x, y):
        output = self.model(x)
        adv = []
        iter0 = []
        iter1 = []
        adv.append(x)
        iter0.append(0)
        iter1.append(0)
        return adv, iter0, iter1, 0

    def target(self, x, y):
        output = self.model(x)
        adv = []
        iter0 = []
        iter1 = []
        adv.append(x)
        iter0.append(0)
        iter1.append(0)
        return adv, iter0, iter1, 0


def get_attack_algorithm(data_type, device, cfg, model, q_budgets, stop=True):
    alg_type = cfg["name"]
    if data_type == 'cifar10':
        if alg_type == 'none' or alg_type == 'benign':
            alg = NoneMethod(model=model)
        elif alg_type == 'hsja':
            alg = attack.HSJA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'nes':
            alg = attack.NES(device, model=model, q_budgets=q_budgets, eps=255, sigma=0.1, lr=0.55, stop=stop)
        elif alg_type == 'simba':
            alg = attack.SimBA(device, model=model, q_budgets=q_budgets, dct=False, eps=0.03, stop=stop)
        elif alg_type == 'ba':
            alg = attack.BA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'sign-opt':
            alg = attack.SignOpt(device, model=model, q_budgets=q_budgets, eps=255, beta=0.001, stop=stop)
        elif alg_type == 'sign-flip':
            alg = attack.SignFlip(device, model=model, q_budgets=q_budgets, eps=0.1, stop=stop)
        elif alg_type == 'square':
            alg = attack.Square2(device, model=model, q_budgets=q_budgets, eps=0.01, stop=stop)
        else:
            raise Exception("Invalid attack algorithm name")
    elif data_type == 'tiny_imagenet':
        if alg_type == 'none' or alg_type == 'benign':
            alg = NoneMethod(model=model)
        elif alg_type == 'hsja':
            alg = attack.HSJA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'nes':
            alg = attack.NES(device, model=model, q_budgets=q_budgets, eps=255, sigma=0.1, lr=2.55, stop=stop)
        elif alg_type == 'simba':
            alg = attack.SimBA(device, model=model, q_budgets=q_budgets, dct=False, eps=0.2, stop=stop)
        elif alg_type == 'ba':
            alg = attack.BA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'sign-opt':
            alg = attack.SignOpt(device, model=model, q_budgets=q_budgets, eps=5, beta=0.05, stop=stop)
        elif alg_type == 'sign-flip':
            alg = attack.SignFlip(device, model=model, q_budgets=q_budgets, eps=0.1, stop=stop)
        elif alg_type == 'square':
            alg = attack.Square2(device, model=model, q_budgets=q_budgets, eps=0.01, stop=stop)
        else:
            raise Exception("Invalid attack algorithm name")
    elif data_type == 'imagenet':
        if alg_type == 'none' or alg_type == 'benign':
            alg = NoneMethod(model=model)
        elif alg_type == 'hsja':
            alg = attack.HSJA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'nes':
            alg = attack.NES(device, model=model, q_budgets=q_budgets, eps=255, sigma=0.1, lr=2.55, stop=stop)
        elif alg_type == 'simba':
            alg = attack.SimBA(device, model=model, q_budgets=q_budgets, dct=False, eps=0.2, stop=stop)
        elif alg_type == 'ba':
            alg = attack.BA(device, model=model, q_budgets=q_budgets, stop=stop)
        elif alg_type == 'sign-opt':
            alg = attack.SignOpt(device, model=model, q_budgets=q_budgets, eps=5, beta=0.05, stop=stop)
        elif alg_type == 'sign-flip':
            alg = attack.SignFlip(device, model=model, q_budgets=q_budgets, eps=0.1, stop=stop)
        elif alg_type == 'square':
            alg = attack.Square2(device, model=model, q_budgets=q_budgets, eps=0.01, stop=stop)
        else:
            raise Exception("Invalid attack algorithm name")
    else:
        raise Exception("Invalid dataset type")
    return alg

class Method:
    def __init__(self, data_type, device, cfg, d_model=None, q_budgets=[10000], stop=True):
        super().__init__()
        self.data_type = data_type
        self.device = device
        self.model = d_model
        self.alg = self.algorithm(data_type, device, cfg, d_model, q_budgets, stop)

    def algorithm(self, data_type, device, cfg, model, q_budgets, stop):
        alg = get_attack_algorithm(data_type, device, cfg, model, q_budgets, stop)
        return alg

    def run(self, x, y):
        x = x.clone().detach()
        x_adv, iter0, iter1, iter2 = self.alg.untarget(x, y)
        return x_adv, iter0, iter1, iter2



