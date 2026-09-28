import pytest

from zvcfd.perfmodel import GPUS, METHODS, cpu_fv_estimate, estimate


def test_memory_follows_stored_cells():
    e = estimate(1e9, fill=0.5, gpus=8)
    assert e.stored_cells == pytest.approx(2e9)
    assert e.memory_per_gpu_gb == pytest.approx(2e9 * METHODS["lbm-fp32"].bytes_per_cell / 8 / 1e9)


def test_dense_needs_box():
    with pytest.raises(ValueError):
        estimate(1e9, layout="dense")
    e = estimate(2e8, layout="dense", box_cells=1e9, geometry="porous", gpus=1)
    assert e.stored_cells == 1e9


def test_fp16_is_smaller_and_fits_more():
    a = estimate(3e9, gpus=8, method="lbm-fp32")
    b = estimate(3e9, gpus=8, method="lbm-fp16")
    assert b.memory_per_gpu_gb < a.memory_per_gpu_gb
    assert not a.fits and b.fits


def test_rate_scales_with_bandwidth():
    h = estimate(1e8, gpus=1, gpu="H100-SXM")
    a = estimate(1e8, gpus=1, gpu="RTX-A2000")
    ratio = GPUS["H100-SXM"].copy_gbs / GPUS["RTX-A2000"].copy_gbs
    assert h.updates_per_s / a.updates_per_s == pytest.approx(ratio)


def test_cpu_fv():
    assert cpu_fv_estimate(1.2e7, 1000, cores=100) == pytest.approx(1000.0)
