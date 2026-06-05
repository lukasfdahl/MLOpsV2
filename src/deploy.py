import os
import sys
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone

MODEL_NAME = "CustomCNN"
MODEL_CARD_PATH = os.environ.get("MODEL_CARD_PATH", "model_card.yaml")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def load_model_card() -> dict:
    with open(MODEL_CARD_PATH) as f:
        return yaml.safe_load(f)


def save_model_card(card: dict) -> None:
    with open(MODEL_CARD_PATH, "w") as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)


def get_latest_model_version(client: MlflowClient) -> mlflow.entities.model_registry.ModelVersion | None:
    """Return the most recently created version of the registered model."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"Could not fetch model versions: {e}")
        return None

    if not versions:
        print(f"No versions found for model '{MODEL_NAME}'")
        return None

    # Sort by creation timestamp descending
    return sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]


def get_run_metrics(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    return run.data.metrics


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    card = load_model_card()
    threshold = card["evaluation"]["deployment_threshold"]["val_loss"]

    version = get_latest_model_version(client)
    if version is None:
        print("No model to evaluate — aborting deployment.")
        return 1

    metrics = get_run_metrics(client, version.run_id)
    val_loss = metrics.get("val_loss")
    val_acc = metrics.get("val_acc")

    print(f"Latest model version : {version.version}")
    print(f"Run ID               : {version.run_id}")
    print(f"val_loss             : {val_loss}")
    print(f"val_acc              : {val_acc}")
    print(f"Deployment threshold : val_loss < {threshold}")

    if val_loss is None:
        print("val_loss metric missing from run — cannot evaluate. Aborting.")
        return 1

    if val_loss < threshold:
        print(f"\nCriteria met (val_loss={val_loss:.4f} < {threshold}). Promoting to Production.")

        # Archive any existing Production version first
        existing_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for prod_ver in existing_prod:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=prod_ver.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"  Archived previous Production version {prod_ver.version}")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version.version,
            stage="Production",
            archive_existing_versions=False,
        )

        deployed_at = datetime.now(timezone.utc).isoformat()
        client.set_model_version_tag(MODEL_NAME, version.version, "deployed_at", deployed_at)
        client.set_model_version_tag(MODEL_NAME, version.version, "deployment_status", "Production")

        # Update model card with latest performance info
        card["performance"]["best_val_loss"] = round(float(val_loss), 6)
        card["performance"]["best_val_acc"] = round(float(val_acc), 4) if val_acc else None
        card["performance"]["run_id"] = version.run_id
        card["performance"]["trained_at"] = deployed_at
        save_model_card(card)

        # Log updated model card and deployment event to the training run
        with mlflow.start_run(run_id=version.run_id):
            mlflow.log_artifact(MODEL_CARD_PATH, artifact_path="model_card")
            mlflow.log_param("deployment_status", "Production")
            mlflow.log_param("deployed_at", deployed_at)

        print(f"Model card updated and logged to MLflow run {version.run_id}")
        print("Deployment complete.")
        return 0

    else:
        print(f"\nCriteria NOT met (val_loss={val_loss:.4f} >= {threshold}). Model NOT deployed.")

        client.set_model_version_tag(MODEL_NAME, version.version, "deployment_status", "Rejected")

        with mlflow.start_run(run_id=version.run_id):
            mlflow.log_param("deployment_status", "Rejected")
            mlflow.log_param("rejection_reason", f"val_loss={val_loss:.4f} >= threshold={threshold}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
