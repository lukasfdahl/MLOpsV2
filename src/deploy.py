# deploy.py
import os
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "CustomCNN"
VAL_LOSS_THRESHOLD = 7.5

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
)
client = MlflowClient()


def deploy_latest_model():
    """
    Promote the latest version of the registered model to Production in MLflow
    if it meets the performance threshold, and log the deployment as a run.
    """
    # Get the latest model version
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
    if not versions:
        print(f"ERROR: No versions found for model '{REGISTERED_MODEL_NAME}'")
        print("Run evaluate.py first to register the model.")
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    print(f"Latest model: '{REGISTERED_MODEL_NAME}' version {latest.version}")

    # Fetch the val_loss from the training run that produced this model
    run = client.get_run(latest.run_id)
    val_loss = run.data.metrics.get("val_loss")

    if val_loss is None:
        print("WARNING: No val_loss found in run metrics, deploying anyway.")
    elif val_loss >= VAL_LOSS_THRESHOLD:
        print(f"Model did NOT pass threshold (val_loss={val_loss:.4f} >= {VAL_LOSS_THRESHOLD}) — skipping deployment.")
        return

    # Promote to Production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model version {latest.version} promoted to Production.")

    # Log the deployment as an MLflow run
    mlflow.set_experiment("custom_model_deployment")
    with mlflow.start_run(run_name="deployment"):
        mlflow.log_params({
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": latest.version,
            "stage": "Production",
        })
        if val_loss is not None:
            mlflow.log_metric("deployed_val_loss", val_loss)

        print(f"Deployment logged to MLflow.")


if __name__ == "__main__":
    deploy_latest_model()
