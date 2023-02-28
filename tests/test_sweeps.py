import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "aics_im2im/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        "-m",
        "experiment=im2im/omnipose",
        "data=test/omnipose",
        "model=test/omnipose",
        "hydra.sweep.dir=" + str(tmp_path),
        "+model.optimizer.lr=0.005,0.01",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep_ddp_sim(tmp_path):
    """Test default hydra sweep with ddp sim."""
    command = [
        startfile,
        "-m",
        "experiment=im2im/omnipose",
        "data=test/omnipose",
        "model=test/omnipose",
        "hydra.sweep.dir=" + str(tmp_path),
        "trainer=ddp_sim",
        "trainer.max_epochs=3",
        "+trainer.limit_train_batches=2",
        "+trainer.limit_val_batches=2",
        "+trainer.limit_test_batches=2",
        "+model.optimizer.lr=0.005,0.01,0.02",
    ] + overrides
    run_sh_command(command)
