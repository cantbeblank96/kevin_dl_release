import os
from kevin_dl.workers.transforms import Base_Transform
from kevin_toolbox.patches.for_test import check_consistency


class Test_Transform(Base_Transform):
    name = ":114:514"

    def cal(self, input_s, alpha=None, beta=None, **kwargs):
        """
            参数：
                input_s:                <dict>
                                            其中包含：
                                                image:      <torch.tensor> shape [C, H, W]
                ...                     用于构建 worker 的参数
        """
        assert alpha is not None and beta is not None
        input_s["y"] = input_s["x"] * alpha + beta

        return input_s


def test_call():
    print("test Base_Transform.__call__()")

    ts = Test_Transform(b_include_details=True, alpha=1, beta=2)
    ts_2 = Test_Transform(b_include_details=True, alpha=3, beta=4)
    output_s = ts(input_s=dict(x=3))
    check_consistency(
        output_s,
        {
            'x': 3, 'y': 5,
            'details_ls': [{'name': Test_Transform.name, 'paras': {'alpha': 1, 'beta': 2}}],
            'details': {'name': Test_Transform.name, 'paras': {'alpha': 1, 'beta': 2}}
        }
    )
    output_s = ts_2(output_s)
    check_consistency(
        output_s,
        {
            'x': 3, 'y': 13,
            'details_ls': [{'name': Test_Transform.name, 'paras': {'alpha': 1, 'beta': 2}},
                           {'name': Test_Transform.name, 'paras': {'alpha': 3, 'beta': 4}}],
            'details': {'name': Test_Transform.name, 'paras': {'alpha': 3, 'beta': 4}}
        }
    )


def test_random_item_in_paras():
    print("test Base_Transform.find_random_item_in_paras()/determine_paras()")

    ts = Test_Transform(
        b_include_details=True,
        alpha={
            "p_type": "categorical",
            "choices": [1, 2, 4]
        },
        beta={
            "p_type": "float",
            "p_prob": "normal",
            "high": 0.2,
            "low": -0.2,
        }
    )
    output_s = ts(input_s=dict(x=3))
    assert output_s['details']['paras']['alpha'] in [1, 2, 4]
    assert -0.2 <= output_s['details']['paras']['beta'] < 0.2


def test_replay_last_process():
    print("test Base_Transform.replay_last_process()")

    ts = Test_Transform(
        b_include_details=True,
        replay_times=2,
        alpha={
            "p_type": "float",
            "p_prob": "normal",
            "high": 1,
            "low": -1,
        },
        beta={
            "p_type": "float",
            "p_prob": "normal",
            "high": 0.2,
            "low": -0.2,
        }
    )

    # 前 replay_times 次保持不变
    output_s = ts(input_s=dict(x=3))
    output_s_1 = ts(input_s=dict(x=4))
    # 即使临时改了参数范围，还是不变
    ts.paras['alpha'].update({'high': 10, 'low': 5})
    output_s_2 = ts(input_s=dict(x=5))
    check_consistency(*[i['details'] for i in [output_s, output_s_1, output_s_2]])

    # 后面将发生变化
    output_s_3 = ts(input_s=dict(x=6))
    try:
        check_consistency(*[i['details'] for i in [output_s, output_s_3]])
        assert False
    except:
        assert True

    # 重设 replay_times 为 -1，将一直保持重播
    ts.replay_last_process(replay_times=-1)
    #
    output_s_ls = []
    ts.paras['alpha'].update({'high': 1, 'low': -1})
    while len(output_s_ls) < 5:
        output_s_ls.append(ts(input_s=dict(x=3)))
    check_consistency(*[i['details'] for i in output_s_ls])

    # 重设 replay_times 为 None 时，将取消重播
    ts.replay_last_process(replay_times=None)
    output_s = ts(input_s=dict(x=6))
    try:
        check_consistency(*[i['details'] for i in [output_s, output_s_ls[-1]]])
        assert False
    except:
        assert True


def test_set_rng():
    print("test Base_Transform.set_rng()")

    ts = Test_Transform(
        b_include_details=True,
        seed=114514,
        alpha={
            "p_type": "float",
            "p_prob": "normal",
            "high": 1,
            "low": -1,
        },
        beta={
            "p_type": "float",
            "p_prob": "normal",
            "high": 0.2,
            "low": -0.2,
        }
    )

    output_s_ls = []
    while len(output_s_ls) < 5:
        output_s_ls.append(ts(input_s=dict(x=3)))
    try:
        check_consistency(*[i['details'] for i in output_s_ls])
        assert False
    except:
        assert True

    # 重设 seed
    ts.set_rng(seed=114514)
    output_s = ts(input_s=dict(x=6))
    check_consistency(*[i['details'] for i in [output_s, output_s_ls[0]]])
