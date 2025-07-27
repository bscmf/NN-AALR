from torch.optim import Optimizer
import torch
import copy


class SVRG_AALR(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0, varphi=2e-5, steps=400, gamma=1e-2):
        self.u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if steps < 0.0:
            raise ValueError("Invalid step size: {}".format(steps))
        if gamma < 0.0:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if varphi < 0.0:
            raise ValueError("Invalid varphi: {}".format(varphi))

        defaults = dict(lr=lr, weight_decay=weight_decay, steps=int(steps), gamma=gamma)

        super(SVRG_AALR, self).__init__(params, defaults)
        self._params = self.param_groups[0]["params"]

    def get_param_groups(self):
        return self.param_groups

    def set_u(self, new_u):
        if self.u is None:
            self.u = copy.deepcopy(new_u)
        for u_group, new_group in zip(self.u, new_u):
            for u, new_u in zip(u_group["params"], new_group["params"]):
                u.grad = new_u.grad.clone()

    def step(self, params):
        group = self.param_groups[0]
        state = self.state[self._params[0]]
        state.setdefault("bb_iter", -1)
        state.setdefault("n_iter", -1)

        state["n_iter"] += 1
        if state["n_iter"] % group["steps"] == 0:
            state["bb_iter"] += 1
            sum_dp_dg = 0
            sum_dp_norm = 0
            sum_dg_norm = 0
            delta = 0
            grads_prev = 0

            for p in self._params:
                if state["n_iter"] == 0:
                    with torch.no_grad():
                        self.state[p]["grad_aver"] = torch.zeros_like(p)
                        self.state[p]["grads_prev"] = torch.zeros_like(p)
                        self.state[p]["params_prev"] = torch.zeros_like(p)

                if state["bb_iter"] > 1:
                    params_diff = p.detach() - self.state[p]["params_prev"]
                    grads_diff = self.state[p]["grad_aver"] - self.state[p]["grads_prev"]
                    sum_dp_dg += (grads_diff * params_diff).sum().item()
                    sum_dp_norm += params_diff.norm().item() ** 2
                    sum_dg_norm += grads_diff.norm().item() ** 2
                    delta = params_diff.norm().item()
                    grads_prev = self.state[p]["grads_prev"].norm().item()

                if state["bb_iter"] > 0:
                    self.state[p]["grads_prev"].copy_(self.state[p]["grad_aver"])
                    self.state[p]["params_prev"].copy_(p.detach())
                    self.state[p]["grad_aver"].zero_()

            if state["bb_iter"] > 1:
                if abs(sum_dp_dg) >= 1e-10:
                    lr_bb1 = abs(sum_dp_norm / (sum_dp_dg * group["steps"]))
                    lr_bb2 = abs(sum_dp_dg / (sum_dg_norm * group["steps"]))

                    pre_lr = group["lr"]
                    phi = min(1, max(0, abs((pre_lr - lr_bb1) / (lr_bb1 - lr_bb2))))

                    lr_mbb = phi * lr_bb1 + (1 - phi) * lr_bb2
                    if state["bb_iter"] % 2 == 0:
                        lr_tmp = lr_bb1
                    else:
                        lr_tmp = lr_mbb
                    lr = max(group["varphi"] / (state["bb_iter"] + 1), min(lr_tmp, abs(delta / grads_prev)))
                    group["lr"] = lr

        for group, new_group, u_group in zip(self.param_groups, params, self.u):
            weight_decay = group["weight_decay"]
            for p, q, u in zip(group["params"], new_group["params"], u_group["params"]):
                if p.grad is None:
                    continue
                if q.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                v = d_p - q.grad.data + u.grad.data
                p.data.add_(-group["lr"], v)

                with torch.no_grad():
                    self.state[p]["grad_aver"].mul_(1 - group["gamma"]).add_(group["gamma"], v)


class SVRG_AALR_TMP(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(SVRG_AALR_TMP, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups

    def set_param_groups(self, new_params):
        for group, new_group in zip(self.param_groups, new_params):
            for p, q in zip(group["params"], new_group["params"]):
                p.data[:] = q.data[:]
