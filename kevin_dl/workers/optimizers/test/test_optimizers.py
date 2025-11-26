import torch
import numpy as np
from kevin_toolbox.patches.for_test import check_consistency
from kevin_toolbox.computer_science.algorithm.scheduler import Trigger
from kevin_dl.workers.variable import OPTIMIZERS
from kevin_dl.workers.optimizers import Advanced_Optimizer, get_param_groups_from_settings
from kevin_dl.workers.optimizers.test.test_data.example_model import Example_Model


def test_get_param_groups_from_settings():
    print("test optimizers.get_param_groups_from_settings()")

    res = get_param_groups_from_settings(
        settings={
            "for_all": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0},
            "for_groups": {
                "vc_coeff": {"lr": 1e-3, "weight_decay": 0}
            },
        },
        named_parameters=[("model.conv2d.weight", torch.ones([3, 3], requires_grad=True)),
                          ("model.conv2d.bias", torch.ones([1, 3], requires_grad=False)),
                          ("model.vc_coeff.weight", torch.zeros([4, 4], requires_grad=True))]
    )

    check_consistency([np.ones([3, 3])], [i.detach().numpy() for i in res[0].pop("params")])
    check_consistency([np.zeros([4, 4])], [i.detach().numpy() for i in res[1].pop("params")])
    check_consistency(
        [{'lr': 1e-2, 'weight_decay': 1e-4, 'momentum': 0}, {'lr': 1e-3, 'weight_decay': 0, 'momentum': 0}, ],
        res
    )


def test_Advanced_Optimizer():
    print("test optimizers.Advanced_Optimizer")

    net = Example_Model(in_channels=3, out_channels=12, n_output=10)
    # for i, j in net.named_parameters():
    #     print(i, j.shape, j.requires_grad)
    # hidden.0.weight torch.Size([6, 3, 3, 3]) True
    # hidden.1.weight torch.Size([6]) True
    # hidden.1.bias torch.Size([6]) True
    # hidden.2.weight torch.Size([12, 6, 5, 5]) True
    # predict.weight torch.Size([10, 12]) True
    # predict.bias torch.Size([10]) True

    opt = Advanced_Optimizer(
        builder=OPTIMIZERS.get(name=":torch:optim:Adam"),
        named_parameters=net.named_parameters(),
        settings=dict(for_all=dict(lr=1e-3, betas=[0.9, 0.999]), for_groups={"bias": dict(lr=0)}),
        # 策略
        strategy=[
            {
                # bias
                "__dict_form": "trigger_value:para_name",
                "__trigger_name": "epoch",
                "<eval>lambda x: x%100==0": {
                    ":settings:for_groups:bias:lr": 0.1,
                },
            },
            {
                # for_all
                "__dict_form": "trigger_value:para_name",
                "__trigger_name": "epoch",
                "<eval>lambda x: x%300==0": {
                    ":settings:for_all:betas@0": "<eval>lambda p, t: round(p*0.1,3)",
                    ":settings:for_all:lr": 0.1
                },
            }
        ]
    )
    trigger = Trigger(target_s={":opt": opt})

    def get_param_groups(opt):
        return [{k: v for k, v in i.items() if k in ["lr", "betas", "weight_decay", "params"]} for i in
                opt.state_dict()["param_groups"]]

    #
    expected = [{'lr': 0.001, 'betas': [0.9, 0.999], 'weight_decay': 0, 'params': [0, 1, 2, 3]},
                {'lr': 0, 'betas': [0.9, 0.999], 'weight_decay': 0, 'params': [4, 5]}]
    check_consistency(expected, get_param_groups(opt))

    #
    trigger.update_by_state(cur_state=dict(epoch=100, step=1))
    expected = [{'lr': 0.001, 'betas': [0.9, 0.999], 'weight_decay': 0, 'params': [0, 1, 2, 3]},
                {'lr': 0.1, 'betas': [0.9, 0.999], 'weight_decay': 0, 'params': [4, 5]}]
    check_consistency(expected, get_param_groups(opt))

    trigger.update_by_state(cur_state=dict(epoch=300, step=1))
    expected = [{'lr': 0.1, 'betas': [0.09, 0.999], 'weight_decay': 0, 'params': [0, 1, 2, 3]},
                {'lr': 0.1, 'betas': [0.09, 0.999], 'weight_decay': 0, 'params': [4, 5]}]
    check_consistency(expected, get_param_groups(opt))
