from kevin_dl.workers.optimizers import Advanced_Optimizer

if __name__ == '__main__':
    import torch
    from kevin_dl.workers.algorithms.variational_conv.test.example_model import Net

    net = Net(in_channels=3, out_channels=12, n_output=10, mode="variational")

    opt = Advanced_Optimizer(
        builder=torch.optim.Adam, named_parameters=net.named_parameters(),
        settings={
            "for_all": {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 0.0001
            }
        },
        strategy={
            "__dict_form": "para_name:trigger_value",
            "__trigger_name": "epoch",
            ":settings:for_all:lr": {
                "<eval>lambda t: 0<=t<=10": {
                    "name": ":lr_scheduler:Linear_LR",
                    "T_max": 10,
                    # "base_lr": 0.1,
                    "start_factor": 0.1,
                    "end_factor": 1.0
                },
                "<eval>lambda t: 10<t<=200": {
                    "name": ":lr_scheduler:Cosine_Annealing_LR",
                    "t_offset": 11,
                    "T_max": 190,
                    # "eta_max": 0.1,
                    "eta_min": 0.0
                },
                "<eval>lambda t: 200<t<=300": {
                    "name": ":lr_scheduler:Cosine_Annealing_Warm_Restarts_LR_Scheduler",
                    "T_0": 50,
                    "T_mult": 0.5,
                    "eta_max": 0.1,
                    "eta_min": 0.0
                },
            }
        }
    )
    print(opt.zero_grad)
    opt.param_groups = []
    print(opt.param_groups)
    # 可视化
    import matplotlib.pyplot as plt

    eta_start = 0.05
    eta_end = 1.0
    T_max = 40

    lrs = []
    for t in range(350):
        opt.update_by_state(trigger_state={"epoch": t})
        opt.update_by_state(trigger_state={"epoch": t})
        lrs.append(opt.paras["settings"]["for_all"]["lr"])

    plt.plot(lrs)
    plt.title("Linear LR Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()
