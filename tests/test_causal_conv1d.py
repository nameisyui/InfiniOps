import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import Payload, get_stream


def make_inputs(length, channels, nonzero_state=True, has_bias=True):
    rng = torch.Generator().manual_seed(20261019)
    x = torch.randn(1, length, channels, generator=rng)
    weight = torch.randn(channels, 1, 4, generator=rng)
    bias = torch.randn(channels, generator=rng) if has_bias else None
    state = (
        torch.randn(1, channels, 3, generator=rng)
        if nonzero_state
        else torch.zeros(1, channels, 3)
    )
    return x, state, weight, bias

def assert_output(actual, expected, label):
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


def run_segments(inputs, lengths, alias, implementation_index):
    x, history, weight, bias = inputs
    state = history.clone()
    initial_pointer = state.data_ptr()
    outputs = []
    offset = 0
    for length in lengths:
        chunk = x.narrow(1, offset, length).contiguous()
        out = torch.empty_like(chunk)
        final_state = state if alias else torch.empty_like(state)
        infini.ops.causal_conv1d(
            chunk,
            weight,
            state,
            bias,
            None,
            None,
            None,
            out,
            final_state,
            implementation_index=implementation_index,
            stream=get_stream(x.device),
        )
        outputs.append(out)
        if not alias:
            previous = torch.cat((history, x[:, :offset].transpose(1, 2)), dim=-1)[
                ..., -3:
            ]
            torch.testing.assert_close(state, previous, atol=0, rtol=0)
        offset += length
        expected_history = torch.cat((history, x[:, :offset].transpose(1, 2)), dim=-1)[
            ..., -3:
        ]
        torch.testing.assert_close(final_state, expected_history, atol=0, rtol=0)
        state = final_state
    assert offset == x.shape[1]
    if alias:
        assert state.data_ptr() == initial_pointer
    return torch.cat(outputs, dim=1), state


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("channels", [32, 4096])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 17, 128])
@pytest.mark.parametrize("nonzero_state", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_causal_conv1d(
    channels, length, nonzero_state, has_bias, alias, device, implementation_index
):
    inputs = tuple(
        t.to(device) if t is not None else None
        for t in make_inputs(length, channels, nonzero_state, has_bias)
    )
    expected = _reference(*inputs).cpu()
    whole, whole_state = run_segments(inputs, [length], alias, implementation_index)
    expected_output = expected[: whole.numel()].reshape(whole.shape)
    expected_state = expected[whole.numel() :].reshape(whole_state.shape)
    assert_output(whole.cpu(), expected_output, "whole output")
    torch.testing.assert_close(whole_state.cpu(), expected_state, atol=0, rtol=0)
    split = min(2, length)
    for name, lengths in (("split", [split, length - split]), ("tokens", [1] * length)):
        out, state = run_segments(
            inputs, [n for n in lengths if n], alias, implementation_index
        )
        assert_output(out.cpu(), expected_output, f"{name} output")
        assert_output(out.cpu(), whole.cpu(), f"{name} vs whole output")
        torch.testing.assert_close(state, whole_state, atol=0, rtol=0)

    def actual(x, state, weight, bias):
        out, final_state = run_segments(
            (x, state, weight, bias), [length], alias, implementation_index
        )
        assert torch.isfinite(out).all()
        return torch.cat((out.flatten(), final_state.flatten()))

    return Payload(actual, _reference, inputs, {}, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("case", ["history", "impulse"])
def test_conv_weight_order_and_no_activation(alias, case, device, implementation_index):
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(2, 1).view(2, 1, 4)
    if case == "history":
        x = torch.tensor(
            [[[4.0, -4.0], [5.0, -5.0], [6.0, -6.0], [7.0, -7.0], [8.0, -8.0]]]
        )
        history = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]])
        expected = torch.tensor(
            [
                [
                    [30.0, -30.0],
                    [40.0, -40.0],
                    [50.0, -50.0],
                    [60.0, -60.0],
                    [70.0, -70.0],
                ]
            ]
        )
    else:
        x = torch.zeros(1, 5, 2)
        x[0, 0] = torch.tensor([1.0, -1.0])
        history = torch.zeros(1, 2, 3)
        expected = torch.tensor(
            [[[4.0, -4.0], [3.0, -3.0], [2.0, -2.0], [1.0, -1.0], [0.0, 0.0]]]
        )
    inputs = tuple(t.to(device) for t in (x, history, weight)) + (None,)
    out, _ = run_segments(inputs, [1, 1, 3], alias, implementation_index)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("alias", [False, True])
def test_conv_non_default_stream_and_changed_data(alias, device, implementation_index):
    if device != "musa":
        pytest.skip("requires MUSA")

    inputs = make_inputs(2, 32)
    x, state, weight, bias = [t.to(device) for t in inputs]
    base_x, base_state = x.clone(), state.clone()
    out = torch.empty_like(x)
    final_state = state if alias else torch.empty_like(state)
    torch.musa.synchronize()
    stream = torch.musa.Stream()
    assert stream.musa_stream != torch.musa.default_stream().musa_stream
    pointers = (x.data_ptr(), state.data_ptr(), out.data_ptr(), final_state.data_ptr())
    torch.musa._sleep(50_000_000)

    try:
        for scale in (0.5, -0.75):
            changed = (inputs[0] * scale, inputs[1] + scale, inputs[2], inputs[3])

            with torch.musa.stream(stream):
                x.copy_(base_x).mul_(scale)
                state.copy_(base_state).add_(scale)
                infini.ops.causal_conv1d(
                    x,
                    weight,
                    state,
                    bias,
                    None,
                    None,
                    None,
                    out,
                    final_state,
                    implementation_index=implementation_index,
                    stream=get_stream(x.device),
                )
                actual_out, actual_state = out.cpu(), final_state.cpu()
                initial_state = state.cpu()

            expected = _reference(*changed)
            assert_output(
                actual_out, expected[: out.numel()].reshape(out.shape), "stream output"
            )
            torch.testing.assert_close(
                actual_state,
                expected[out.numel() :].reshape(state.shape),
                atol=0,
                rtol=0,
            )

            if not alias:
                torch.testing.assert_close(initial_state, changed[1], atol=0, rtol=0)

            assert (
                x.data_ptr(),
                state.data_ptr(),
                out.data_ptr(),
                final_state.data_ptr(),
            ) == pointers
    finally:
        torch.musa.synchronize()


@pytest.mark.parametrize(
    "mode",
    [
        "cu_seqlens",
        "initial_state_indices",
        "final_state_indices",
        "no_final_state",
        "batch",
        "dtype",
        "layout",
        "kernel",
        "empty",
    ],
)
def test_conv_unsupported_modes(mode, device, implementation_index):
    if device != "musa":
        pytest.skip("requires MUSA")

    x, state, weight, bias = [t.to(device) for t in make_inputs(2, 32)]
    options = [None, None, None]

    if mode in ("cu_seqlens", "initial_state_indices", "final_state_indices"):
        index = ("cu_seqlens", "initial_state_indices", "final_state_indices").index(
            mode
        )
        options[index] = torch.tensor([0, 2] if index == 0 else [0], device=device)
    elif mode == "batch":
        x, state = x.repeat(2, 1, 1), state.repeat(2, 1, 1)
    elif mode == "dtype":
        x, state, weight, bias = [t.half() for t in (x, state, weight, bias)]
    elif mode == "layout":
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    elif mode == "kernel":
        weight, state = weight[..., :3].contiguous(), state[..., :2].contiguous()
    elif mode == "empty":
        x = x[:, :0]

    final_state = (
        None
        if mode in ("no_final_state", "final_state_indices")
        else torch.empty_like(state)
    )

    with pytest.raises(ValueError, match="does not support"):
        infini.ops.causal_conv1d(
            x,
            weight,
            state,
            bias,
            *options,
            torch.empty_like(x),
            final_state,
            implementation_index=implementation_index,
            stream=get_stream(x.device),
        )


def _reference(x, state, weight, bias):
    target = x.device
    x, state, weight = [t.cpu() for t in (x, state, weight)]
    bias = bias.cpu() if bias is not None else None
    extended = torch.cat((state, x.transpose(1, 2)), dim=-1)
    out = F.conv1d(extended, weight, bias, groups=x.shape[-1]).transpose(1, 2)
    return torch.cat((out.flatten(), extended[..., -3:].flatten())).to(target)
