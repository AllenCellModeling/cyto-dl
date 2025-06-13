import os
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union
import threading
import time
import psutil
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    NVMLError,
)
import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger as _MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.file_utils import local_file_uri_to_path
from omegaconf import OmegaConf
from cyto_dl import utils

log = utils.get_pylogger(__name__)


class MLFlowLogger(_MLFlowLogger):
    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        prefix: str = "",
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        fault_tolerant=True,
    ):
        _MLFlowLogger.__init__(
            self,
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
        )

        self.fault_tolerant = fault_tolerant

        if tracking_uri is not None:
            log.info(f"Setting tracking URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
            
        log.info(f"Creating or fetching experiment: {experiment_name}")
        try: 
            mlflow.create_experiment(experiment_name)
        except:
            pass
        mlflow.set_experiment(experiment_name)

        log.info('done setting experiment')

    def start_system_metrics_logging(self, interval: int = 5):
        """
        Starts a background thread that logs CPU, memory, and GPU usage to MLflow every `interval` seconds.
        """
        def log_metrics_loop():
            try:
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
            except NVMLError:
                log.warning("Unable to initialize NVML for GPU monitoring.")
                return

            while getattr(threading.current_thread(), "keep_running", True):
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent

                try:
                    gpu_mem = nvmlDeviceGetMemoryInfo(handle)
                    gpu_used = gpu_mem.used // (1024 * 1024)  # MB
                except NVMLError:
                    gpu_used = 0

                try:
                    self.experiment.log_metrics(
                        self.run_id,
                        {
                            "system/cpu_percent": cpu,
                            "system/mem_percent": mem,
                            "system/gpu_mem_used_mb": gpu_used,
                        },
                        step=int(time.time()),
                    )
                except Exception as e:
                    if self.fault_tolerant:
                        log.warn(f"[MLFlowLogger] Failed to log system metrics: {e}")
                    else:
                        raise e

                time.sleep(interval)

            try:
                nvmlShutdown()
            except Exception:
                pass

        self._sys_metrics_thread = threading.Thread(
            target=log_metrics_loop,
            daemon=True,
        )
        self._sys_metrics_thread.keep_running = True
        self._sys_metrics_thread.start()

    def stop_system_metrics_logging(self):
        """Stops the system metrics thread cleanly."""
        if hasattr(self, "_sys_metrics_thread"):
            self._sys_metrics_thread.keep_running = False
            self._sys_metrics_thread.join(timeout=10)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], mode="train") -> None:
        requirements = params.pop("requirements", [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_path = Path(tmp_dir) / f"{mode}.yaml"
            with conf_path.open("w") as f:
                config = OmegaConf.create(params)
                OmegaConf.save(config=config, f=f)

            reqs_path = Path(tmp_dir) / f"{mode}-requirements.txt"
            reqs_path.write_text("\n".join(requirements))

            self.experiment.log_artifact(self.run_id, local_path=conf_path, artifact_path="config")
            self.experiment.log_artifact(
                self.run_id, local_path=reqs_path, artifact_path="requirements"
            )

    @rank_zero_only
    def log_profiler_artifacts(self, profiler_output_dir: str):
        """
        Logs profiler artifacts (e.g., from PyTorchProfiler) to MLflow under the 'profiler' artifact path.
        """
        if not os.path.isdir(profiler_output_dir):
            log.warn(f"No profiler output found at: {profiler_output_dir}")
            return
        try:
            self.experiment.log_artifacts(self.run_id, local_dir=profiler_output_dir, artifact_path="profiler")
            log.info(f"Profiler artifacts logged from {profiler_output_dir}")
        except Exception as e:
            if self.fault_tolerant:
                log.warn(f"Failed to log profiler artifacts: {e}")
            else:
                raise e

    @rank_zero_only
    def log_metrics(self, metrics, step):
        try:
            super().log_metrics(metrics, step)
        except Exception as e:
            if self.fault_tolerant:
                log.warn(f"`MLFlowLogger.log_metrics` failed with exception {e}\n\nContinuing...")
            else:
                raise e

    def after_save_checkpoint(self, ckpt_callback):
        try:
            self._after_save_checkpoint(ckpt_callback)
        except Exception as e:
            if self.fault_tolerant:
                log.warn(
                    f"`MLFlowLogger.after_save_checkpoint` failed with exception {e}\n\nContinuing..."
                )
            else:
                raise e

    def _after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint."""
        monitor = ckpt_callback.monitor
        if monitor is not None:
            artifact_path = f"checkpoints/{monitor}"
            existing_ckpts = {
                _.path.split("/")[-1]
                for _ in self.experiment.list_artifacts(self.run_id, path=artifact_path)
            }

            top_k_ckpts = {_.split("/")[-1] for _ in ckpt_callback.best_k_models.keys()}

            to_delete = existing_ckpts - top_k_ckpts
            to_upload = top_k_ckpts - existing_ckpts

            run = self.experiment.get_run(self.run_id)
            repository = get_artifact_repository(run.info.artifact_uri)
            for ckpt in to_delete:
                if isinstance(repository, LocalArtifactRepository):
                    _delete_local_artifact(repository, f"checkpoints/{monitor}/{ckpt}")
                elif hasattr(repository, "delete_artifacts"):
                    repository.delete_artifacts(f"checkpoints/{monitor}/{ckpt}")
                else:
                    warnings.warn(
                        "The artifact repository configured for this "
                        "MLFlow server doesn't support deleting artifacts, "
                        "so we're keeping all checkpoints."
                    )

            for ckpt in to_upload:
                self.experiment.log_artifact(
                    self.run_id,
                    local_path=os.path.join(ckpt_callback.dirpath, ckpt),
                    artifact_path=artifact_path,
                )

            # also save the current best model as "best.ckpt"
            filepath = ckpt_callback.best_model_path
            best_path = Path(filepath).with_name("best.ckpt")
            os.link(filepath, best_path)

            self.experiment.log_artifact(
                self.run_id, local_path=best_path, artifact_path=artifact_path
            )

            best_path.unlink()

        else:
            filepath = ckpt_callback.best_model_path
            artifact_path = "checkpoints"

            # mimic ModelCheckpoint's behavior: if `self.save_top_k == 1` only
            # keep the latest checkpoint, otherwise keep all of them.
            if ckpt_callback.save_top_k == 1:
                last_path = Path(filepath).with_name("last.ckpt")
                os.link(filepath, last_path)

                self.experiment.log_artifact(
                    self.run_id, local_path=last_path, artifact_path=artifact_path
                )

                last_path.unlink()
            else:
                self.experiment.log_artifact(
                    self.run_id, local_path=filepath, artifact_path=artifact_path
                )


def _delete_local_artifact(repo, artifact_path):
    artifact_path = Path(
        local_file_uri_to_path(
            os.path.join(repo._artifact_dir, artifact_path)
            if artifact_path
            else repo._artifact_dir
        )
    )
    if artifact_path.is_file():
        artifact_path.unlink()
