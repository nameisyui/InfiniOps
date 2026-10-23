import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import Payload, get_stream


def make_inputs(length, intermediate, nonzero_state=True, nonzero_bias=True):
    rng = torch.Generator().manual_seed(20261019)
    x = torch.randn(1, length, intermediate, generator=rng) * 0.3
    dt = torch.randn(x.shape, generator=rng) - 2
    dt[..., :3] = torch.tensor([-40.0, 0.0, 40.0])
    b = torch.randn(1, length, 16, generator=rng) * 0.3
    c = torch.randn(b.shape, generator=rng) * 0.3
    a_log = torch.arange(1, 17, dtype=torch.float32).log().repeat(intermediate, 1)
    d = torch.randn(intermediate, generator=rng)
    gate = torch.randn(x.shape, generator=rng)
    bias = (
        torch.randn(intermediate, generator=rng)
        if nonzero_bias
        else torch.zeros(intermediate)
    )
    state = (
        torch.randn(1, intermediate, 16, generator=rng)
        if nonzero_state
        else torch.zeros(1, intermediate, 16)
    )
    return x, dt, b, c, a_log, d, gate, bias, state


def assert_close(actual, expected, label):
    assert torch.isfinite(actual).all(), label
    error = (actual - expected).abs()
    nrmse = (actual - expected).double().norm() / expected.double().norm().clamp_min(
        1e-12
    )
    index = torch.unravel_index(error.argmax(), error.shape)
    print(
        f"{label}: max_abs={error.max():.8g} mean_abs={error.mean():.8g} "
        f"NRMSE={nrmse:.8g} max_index={tuple(int(i) for i in index)} "
        f"ref={expected[index]:.8g}"
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert nrmse <= 1e-5, label


def run_segments(inputs, lengths, implementation_index):
    state = inputs[-1].clone()
    state_pointer = state.data_ptr()
    outputs = []
    offset = 0
    for length in lengths:
        chunk = [
            t.narrow(1, offset, length).contiguous() if i in (0, 1, 2, 3, 6) else t
            for i, t in enumerate(inputs)
        ]
        out = torch.empty_like(chunk[0])
        infini.ops.mamba_selective_scan(
            out,
            *chunk[:-1],
            state,
            implementation_index=implementation_index,
            stream=get_stream(out.device),
        )
        outputs.append(out)
        offset += length
    assert offset == inputs[0].shape[1]
    assert state.data_ptr() == state_pointer
    return torch.cat(outputs, dim=1), state


def check_result(actual, expected, label):
    output, state = actual
    expected_output = expected[: output.numel()].reshape(output.shape)
    expected_state = expected[output.numel() :].reshape(state.shape)
    assert_close(output.cpu(), expected_output.cpu(), f"{label} output")
    assert_close(state.cpu(), expected_state.cpu(), f"{label} state")


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("intermediate", [32, 4096])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 7, 33, 128])
@pytest.mark.parametrize("nonzero_state", [False, True])
@pytest.mark.parametrize("nonzero_bias", [False, True])
def test_mamba_selective_scan(
    intermediate, length, nonzero_state, nonzero_bias, device, implementation_index
):
    inputs = tuple(
        t.to(device)
        for t in make_inputs(length, intermediate, nonzero_state, nonzero_bias)
    )
    expected = _reference(*inputs)
    whole = run_segments(inputs, [length], implementation_index)
    check_result(whole, expected, "whole")
    full = torch.cat([t.flatten() for t in whole])
    split = min(3, length)
    for name, lengths in (
        ("split", [split, length - split]),
        ("tokens", [1] * length),
    ):
        actual = run_segments(inputs, [n for n in lengths if n], implementation_index)
        check_result(actual, expected, name)
        check_result(actual, full, f"{name} vs whole")

    def actual(*args):
        result = run_segments(args, [length], implementation_index)
        return torch.cat([t.flatten() for t in result])

    return Payload(actual, _reference, inputs, {}, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("disabled", ["d", "gate"])
def test_scan_disabled_terms(disabled, device, implementation_index):
    inputs = list(make_inputs(4, 32))
    inputs[5 if disabled == "d" else 6].zero_()
    inputs = tuple(t.to(device) for t in inputs)
    actual = run_segments(inputs, [4], implementation_index)
    check_result(actual, _reference(*inputs), f"{disabled}=0")


def test_scan_non_default_stream_and_changed_data(device, implementation_index):
    if device != "musa":
        pytest.skip("requires MUSA")

    inputs = make_inputs(7, 32)
    tensors = [t.to(device) for t in inputs]
    base_x, base_state = tensors[0].clone(), tensors[-1].clone()
    out = torch.empty_like(tensors[0])
    torch.musa.synchronize()
    stream = torch.musa.Stream()
    assert stream.musa_stream != torch.musa.default_stream().musa_stream
    pointers = [t.data_ptr() for t in tensors]
    torch.musa._sleep(50_000_000)

    try:
        for scale in (0.5, -0.75):
            changed = list(inputs)
            changed[0] = inputs[0] * scale
            changed[-1] = inputs[-1] + scale

            with torch.musa.stream(stream):
                tensors[0].copy_(base_x).mul_(scale)
                tensors[-1].copy_(base_state).add_(scale)
                infini.ops.mamba_selective_scan(
                    out,
                    *tensors,
                    implementation_index=implementation_index,
                    stream=get_stream(out.device),
                )
                actual = (out.cpu(), tensors[-1].cpu())

            check_result(actual, _reference(*changed), "stream and changed data")
            assert [t.data_ptr() for t in tensors] == pointers
    finally:
        torch.musa.synchronize()


def _reference(x, dt, b, c, a_log, d, gate, bias, state):
    # Compute the independent reference in CPU FP32, even for a MUSA test.
    target = x.device
    x, dt, b, c, a_log, d, gate, bias, state = [
        t.cpu() for t in (x, dt, b, c, a_log, d, gate, bias, state)
    ]
    state = state.clone()
    delta = F.softplus(dt + bias)
    a = -a_log.exp()
    output = []
    for t in range(x.shape[1]):
        step = delta[:, t, :, None]
        state = (step * a).exp() * state + step * b[:, t, None, :] * x[:, t, :, None]
        y = (state * c[:, t, None, :]).sum(-1) + d * x[:, t]
        output.append(y * F.silu(gate[:, t]))
    result = torch.cat((torch.stack(output, dim=1).flatten(), state.flatten()))
    assert torch.isfinite(result).all()
    return result.to(target)
