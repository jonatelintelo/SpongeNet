import torch
import numpy as np
import os

from torchvision.utils import save_image
from hardware_model import ASICModel
from get_fid import get_fid

from skimage.metrics import structural_similarity as ssim


bitwidth_to_minvalue = {
    32: 2 ** -126,
    16: 2 ** -30,
    8: 2 ** -14,
}


def add_hooks(model, stats):
    """
    Prepare a model for analysis.
    Intercept computation in each leaf node of the network, and collect data
    on the amount of data accessed and computation performed.
    ASSUMPTION: nothing significant happens in modules which contain other
    modules. Only leaf modules are analysed.
    :param model: a torch.nn.Module to be analysed.
    :param stats: a StatsRecorder into which the results will be stored.
    """
    hooks = []

    leaf_nodes = [module for module in model.modules() if len(list(module.children())) == 0]

    stat_fn = record_stats(stats)
    for module in leaf_nodes:
        hook = module.register_forward_hook(stat_fn)
        hooks.append(hook)

    return hooks


def remove_hooks(hooks):
    """
    Remove hooks from a model.
    :param hooks: an Iterable containing hooks to be removed.
    """
    for hook in hooks:
        hook.remove()


class StatsRecorder:
    def __init__(self, bitwidth=32, n_gpus=10):
        # Note: we need to have one variable for each n_gpus. Otherwise we get errors in the hooks
        self.total_input_activations = torch.zeros(n_gpus)
        self.non_zero_input_activations = torch.zeros(n_gpus)
        self.total_output_activations = torch.zeros(n_gpus)
        self.non_zero_output_activations = torch.zeros(n_gpus)
        self.total_parameters = torch.zeros(n_gpus)
        self.non_zero_parameters = torch.zeros(n_gpus)
        self.computations = 0

        if bitwidth not in bitwidth_to_minvalue:
            raise ValueError("Passed bitwidth is not supported")
        self.min_value = bitwidth_to_minvalue[bitwidth]
        self.nonzero_func = lambda x: float(len((x.abs() > self.min_value).nonzero()))

    def __reset__(self):
        del self.total_input_activations, self.non_zero_input_activations
        del self.total_output_activations, self.non_zero_output_activations
        del self.total_parameters, self.non_zero_parameters, self.computations
        self.__init__()


def get_energy_estimate(stats, hw):
    """
    Estimate the energy consumption in picojoules of a given computation on
    given hardware.
    ASSUMPTIONS:
    * Weights are read from DRAM exactly once.
    * Input activations are read from DRAM exactly once.
    * Output activations are written to DRAM exactly once.
    :param stats: a StatsRecorder containing details of the computation.
    :param hw: a HardwareModel containing details of the processor.
    """
    total = 0.0

    if hw.compress_sparse_weights:
        total += hw.memory_cost * stats.non_zero_parameters.sum()
    else:
        total += hw.memory_cost * stats.total_parameters.sum()

    if hw.compress_sparse_activations:
        total += hw.memory_cost * (stats.non_zero_input_activations.sum() +
                                   stats.non_zero_output_activations.sum())
    else:
        total += hw.memory_cost * (stats.total_input_activations.sum() +
                                   stats.total_output_activations.sum())
    compute_fraction = 1.0

    if hw.compute_skip_zero_weights:
        compute_fraction *= (stats.non_zero_parameters.sum() / stats.total_parameters.sum())

    if hw.compute_skip_zero_activations:
        compute_fraction *= (stats.non_zero_input_activations.sum() /
                             stats.total_input_activations.sum())

    total += compute_fraction * stats.computations * hw.compute_cost

    return total


def record_stats(stats):
    """
    Create a forward hook function which will record information about a layer's
    Create a forward hook function which will record information about a layer's
    execution.
    For all module parameters/buffers, in_data and out_data, record:
    * Number of values
    * Number of non-zeros
    Also estimate amount of computation (depends on layer type).
    :param stats: a StatsRecorder to store results in.
    :return: forward hook function.
    """

    def hook_fn(nonzero_func, module, in_data, out_data):
        # Activations are sometimes Tensors, and sometimes tuples of Tensors.
        # Ensure we're always dealing with tuples.
        if isinstance(in_data, torch.Tensor):
            in_data = (in_data,)
        if isinstance(out_data, torch.Tensor):
            out_data = (out_data,)

        # Collect memory statistics.
        for tensor in in_data:
            stats.total_input_activations[tensor.get_device()] += tensor.numel()
            stats.non_zero_input_activations[tensor.get_device()] += nonzero_func(tensor)

        for tensor in out_data:
            stats.total_output_activations[tensor.get_device()] += tensor.numel()
            stats.non_zero_output_activations[tensor.get_device()] += nonzero_func(tensor)

        for tensor in module.buffers():
            stats.total_parameters[tensor.get_device()] += tensor.numel()
            stats.non_zero_parameters[tensor.get_device()] += nonzero_func(tensor)

        for tensor in module.parameters():
            stats.total_parameters[tensor.get_device()] += tensor.numel()
            stats.non_zero_parameters[tensor.get_device()] += nonzero_func(tensor)

        # Collect computation statistics.
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            # One computation per input pixel - window size is chosen adaptively
            # and windows never overlap (?).
            assert len(in_data) == 1
            input_size = in_data[0].numel()
            stats.computations += input_size
        elif isinstance(module, torch.nn.AvgPool2d) or \
                isinstance(module, torch.nn.MaxPool2d):
            # Each output pixel requires computations on a 2D window of input.
            if type(module.kernel_size) == int:
                # Kernel size here can be either a single int for square kernel
                # or a tuple (see
                # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d )
                window_size = module.kernel_size ** 2
            else:
                window_size = module.kernel_size[0] * module.kernel_size[1]

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0].numel()
            stats.computations += output_size * window_size

        elif isinstance(module, torch.nn.Conv2d):
            # Each output pixel requires computations on a 3D window of input.
            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            _, channels, _, _ = in_data[0].size()
            window_size = \
                module.kernel_size[0] * module.kernel_size[1] * channels

            # Not sure which output tensor to use if there are multiple of them.
            assert len(out_data) == 1
            output_size = out_data[0].numel()

            stats.computations += output_size * window_size

        elif isinstance(module, torch.nn.Dropout2d) or isinstance(module, torch.nn.modules.dropout.Dropout):
            # Do nothing - dropout has no effect during inference.
            pass

        elif isinstance(module, torch.nn.Linear):
            # One computation per weight, for each batch element.

            # Not sure which input tensor to use if there are multiple of them.
            assert len(in_data) == 1
            batch = in_data[0].numel() / in_data[0].shape[-1]

            stats.computations += module.weight.numel() * batch

        elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(module,
                                                                                torch.nn.modules.activation.ReLU6):
            # ReLU does a single negation check
            pass

        elif isinstance(module, torch.nn.LayerNorm):
            # You first compute
            pass

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):

            # Accesses to E[x] and Var[x] (all channel size)

            stats.total_parameters += 2 * module.num_features
            stats.non_zero_parameters += \
                nonzero_func(module.running_mean) + \
                nonzero_func(module.running_var)

            # (x-running_mean)/running variance
            # multiply by gamma and beta addition
            stats.computations += 4 * in_data[0].numel()
        # else:
        #    print("Unsupported module type for energy analysis:", type(module))

    return lambda *x: hook_fn(stats.nonzero_func, *x)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def create_labels(c_org, device, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(device))
        return c_trg_list

def get_energy_consumption_fid(dataloader, model, device, c_dim, attributes):
    stats = StatsRecorder()
    hardware = ASICModel(optim=True)
    hardware_worst = ASICModel(optim=False)
    batch_energy_ratios = []
    batch_energy_pj = []
    fake_image_tensors = []
    fake_image_labels = []

    hooks = add_hooks(model, stats)

    with torch.no_grad():
        for batch_index, (x_real, c_org, _) in enumerate(dataloader):
            stats.__reset__()

            x_real = x_real.to(device)

            c_trg = create_labels(c_org, device, c_dim, attributes)
            x_fake = model(x_real, c_trg[0])

            energy_est_avg = get_energy_estimate(stats, hardware)
            energy_est_worst = get_energy_estimate(stats, hardware_worst)
            rs = energy_est_avg / energy_est_worst
            batch_energy_ratios.append(rs)
            batch_energy_pj.append(energy_est_avg)
            fake_image_tensors.append(x_fake)
            fake_image_labels.append(c_trg[0])
            
    remove_hooks(hooks)

    accuracy = get_fid(torch.utils.data.TensorDataset(torch.stack(fake_image_tensors), 
                                                      torch.stack(fake_image_labels)))
    print(x_fake)
    energy_ratio = np.mean(batch_energy_ratios)
    energy_pj = np.mean(batch_energy_pj)
    # dd
    return energy_ratio, energy_pj, accuracy