import sys
import os
from datetime import datetime
import logging
import torch
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.storage as fos
import fiftyone.utils.ultralytics as fou


logger = logging.getLogger("fiftyone.core.collections")



# Set to writable working dir in teams-do pods
TRAIN_ROOT = "/tmp/yolo/"
MODEL_ROOT = os.path.join(TRAIN_ROOT, "models")
DATA_ROOT = os.path.join(TRAIN_ROOT, "data")
PROJECT_ROOT = os.path.join(TRAIN_ROOT, "projects")


class RunYoloInference(foo.Operator):
    @property
    def config(self):
        """
        Defines how the FiftyOne App should display this operator (name,
        label, whether it shows in the operator browser, etc).
        """
        return foo.OperatorConfig(
            name="run-yolo-inference",  # Must match what's in fiftyone.yml
            label="Run YOLO inference",
            icon="batch_prediction",  # Material UI icon, or path to custom icon
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_placement(self, ctx):
        """
        Optional convenience: place a button in the App so the user can
        click to open this operator's input form.
        """
        return types.Placement(
            types.Places.SAMPLES_GRID_SECONDARY_ACTIONS,
            types.Button(
                label="Apply YOLO model",
                icon="batch_prediction",
                prompt=True,  # always show the operator's input prompt
            ),
        )

    def resolve_input(self, ctx):
        """
        Collect the inputs we need from the user. This defines the form
        that appears in the FiftyOne App when the operator is invoked.
        """
        inputs = types.Object()

        inputs.str(
            "det_field",
            required=True,
            label="Detections field",
        )

        # 1) Local filepath to existing YOLOv8 model weights
        inputs.str(
            "weights_path",
            default="gs://voxel51-demo-fiftyone-ai/yolo/yolov8n_finetuned.pt",
            required=True,
            description="Filepath to the YOLOv8 *.pt weights file",
            label="YOLOv8 weights",
        )

        # 2) Confidence threshold
        inputs.float(
            "conf_thresh",
            default=0.05,
            required=True,
            label="Confidence threshold",
        )
        
        # 3) CUDA target device
        inputs.int(
            "target_device_index",
            default=0,
            required=False,
            description="CUDA Device number to train on. Optional, defaults to device cuda:0",
            label="Target CUDA device number",
        )

        return types.Property(
            inputs,
            view=types.View(label="Run inference with YOLO"),
        )

    def execute(self, ctx):
        """ """

        from ultralytics import YOLO

        det_field = ctx.params["det_field"]
        weights_path = ctx.params["weights_path"]
        conf_thresh = ctx.params["conf_thresh"]

        dataset = ctx.dataset

        # --- Step 1: Verify the weights_path is YOLOv8 ---
        local_weights_path = os.path.join(MODEL_ROOT, os.path.basename(weights_path))
        fos.copy_file(weights_path, local_weights_path)
        # model = self._try_load_model(local_weights_path)
        str = f"Model downloaded to: {local_weights_path}"
        logger.warning(str)

        cuda_device_count = torch.cuda.device_count()
        logger.warning(f"Number of CUDA devices found: {cuda_device_count}")

        model = YOLO(local_weights_path)

        if cuda_device_count > 1 and target_device_index <= cuda_device_count:
            target_device = f"cuda:{target_device_index}"
            model.to(target_device)
        else:
            model.to("cuda:0")

        for sample in dataset.iter_samples(progress=True,autosave=True):
            result = model(sample.local_path,conf=conf_thresh)[0]
            sample[det_field] = fou.to_detections(result)

        logger.warning("Ending inference")

        return {"status": "success", "samples_processed": len(dataset)}

    def resolve_output(self, ctx):
        """
        Display any final outputs in the App after training completes.
        """
        outputs = types.Object()
        outputs.str(
            "status",
            label="Inference status",
        )

        outputs.str("samples_processed", label="Number of images processed")
        return types.Property(
            outputs,
            view=types.View(label="Inference Results"),
        )


def register(plugin):
    """
    Called by FiftyOne to discover and register your plugin’s operators/panels.
    """
    plugin.register(RunYoloInference)

