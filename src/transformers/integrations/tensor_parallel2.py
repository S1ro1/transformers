from functools import partial

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module


TP_PLAN_MAPPING = {
    "rowwise": RowwiseParallel(),
    "colwise": ColwiseParallel(),
    "sequence": SequenceParallel(),
    "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
}


def prepare_tp_model(module, device_mesh):
    base_tp_plan = module._tp_plan
    base_config_tp_plan = module.config.base_model_tp_plan

    base_tp_plan = {**base_tp_plan, **base_config_tp_plan}

    if base_tp_plan is None:
        return

    model_tp_plan = {k: TP_PLAN_MAPPING[v] for k, v in base_tp_plan.items() if "layers" not in k}
    layer_tp_plan = {
        k.removeprefix("layers.*."): TP_PLAN_MAPPING[v]
        for k, v in base_tp_plan.items()
        if k.startswith("layers.*.")  # and not isinstance(TP_PLAN_MAPPING[v], RowwiseParallel)
    }

    parallelize_module(
        module,
        device_mesh=device_mesh["tp"],
        parallelize_plan=model_tp_plan,
    )

    for i, layer in enumerate(module.model.layers):
        parallelize_module(
            layer,
            device_mesh=device_mesh["tp"],
            parallelize_plan=layer_tp_plan,
        )
    return module
