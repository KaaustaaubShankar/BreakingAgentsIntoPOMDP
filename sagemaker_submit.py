"""Submit a SageMaker training job for this project.
Defaults pre-filled from Stage 1 scan. Review before running real jobs (spot saves 60-70%).
Usage: python sagemaker_submit.py [--instance ml.g6e.xlarge] [--spot] [--no-wait] [-- <entry-point args>]

Any flags after `--` (or any unknown flags) are forwarded to the entry-point script as
SageMaker hyperparameters, which arrive as `--key value` CLI args inside the container.
Example: `python sagemaker_submit.py --spot -- --provider qwen-local --model Qwen/Qwen3-14B --trials 10`
runs the entry point with `--provider qwen-local --model Qwen/Qwen3-14B --trials 10`.
"""
import argparse, os, shutil, tempfile, time
import boto3
import sagemaker
from sagemaker.estimator import Estimator


SOURCE_WHITELIST = (
    "ka59_game",
    "scripts",
    "environment_files",
    "env3",
    "env4",
    "experiments",
    "tests",
    "pyproject.toml",
    "setup.py",
    "README.md",
)


def stage_source_dir(repo_root: str) -> str:
    """Copy whitelist + swap requirements-sagemaker.txt → requirements.txt.

    SageMaker auto-installs requirements.txt from source_dir at container
    start. We keep the project's requirements.txt API-only (no torch) so
    local devs don't pay GPU install cost, and ship the GPU-augmented
    requirements-sagemaker.txt for cloud runs.
    """
    staging = tempfile.mkdtemp(prefix="sm-source-")
    for name in SOURCE_WHITELIST:
        src = os.path.join(repo_root, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(staging, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", "*.npz", "*.pt", "*.bin", "*.safetensors"))
        else:
            shutil.copy2(src, dst)
    sm_req = os.path.join(repo_root, "requirements-sagemaker.txt")
    if os.path.exists(sm_req):
        shutil.copy2(sm_req, os.path.join(staging, "requirements.txt"))
    elif os.path.exists(os.path.join(repo_root, "requirements.txt")):
        shutil.copy2(os.path.join(repo_root, "requirements.txt"),
                     os.path.join(staging, "requirements.txt"))
    return staging


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", default=os.environ.get("INSTANCE_TYPE", "ml.g6e.xlarge"))
    p.add_argument("--max-runtime-hours", type=int, default=4)
    p.add_argument("--spot", action="store_true", help="use spot instances (60-70%% cheaper; safe if resume-friendly)")
    p.add_argument("--no-wait", action="store_true", help="submit async; poll status with describe-training-job")
    p.add_argument("--job-name", default=None)
    p.add_argument("--entry-point", default="scripts/run_real_ablation.py",
                   help="path inside source_dir for the training script. "
                        "Examples: scripts/run_real_ablation.py (ka59), env3/ablation.py (LS20), "
                        "env4/ablation.py (BP35).")
    args, passthrough = p.parse_known_args()

    hyperparameters = {}
    i = 0
    while i < len(passthrough):
        tok = passthrough[i]
        if tok == "--":
            i += 1
            continue
        if tok.startswith("--"):
            key = tok[2:]
            if i + 1 < len(passthrough) and not passthrough[i + 1].startswith("--"):
                hyperparameters[key] = passthrough[i + 1]
                i += 2
            else:
                hyperparameters[key] = "true"
                i += 1
        else:
            i += 1

    region = os.environ["AWS_REGION"]
    bucket = os.environ["S3_BUCKET"]
    handle = os.environ["STUDENT_HANDLE"]
    project = os.environ["PROJECT_SLUG"]
    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/AmazonSageMaker-{handle}"

    # AWS Deep Learning Container — PyTorch 2.9 / py312 / CUDA 13.0.
    # py312 is REQUIRED: arc-agi>=0.9.6 (used by ka59_game/env3/env4 harnesses)
    # only ships wheels for python >=3.12. Earlier py311 image (2.4.0) failed
    # with InstallRequirementsError on `pip install arc-agi`. Catalog:
    # https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

    job_name = args.job_name or f"{project}-{int(time.time())}"
    staging_dir = stage_source_dir(os.getcwd())
    print(f"Submitting {job_name}")
    print(f"  source:   {staging_dir} (staged from {os.getcwd()}, whitelist only)")
    print(f"  instance: {args.instance}{' (spot)' if args.spot else ' (on-demand)'}")
    print(f"  max run:  {args.max_runtime_hours}h")

    estimator = Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_type=args.instance,
        instance_count=1,
        volume_size=100,
        max_run=args.max_runtime_hours * 3600,
        use_spot_instances=args.spot,
        max_wait=(args.max_runtime_hours * 3600 + 3600) if args.spot else None,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/" if args.spot else None,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region)),
        output_path=f"s3://{bucket}/runs/{job_name}/",
        base_job_name=project,
        entry_point=args.entry_point,
        source_dir=staging_dir,
        hyperparameters=hyperparameters,
        environment={
            **({"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}),
            "S3_BUCKET": bucket,
            "HF_HOME": "/opt/ml/input/data/hf_cache",
            "TRANSFORMERS_CACHE": "/opt/ml/input/data/hf_cache",
        },
        tags=[
            {"Key": "participant", "Value": handle},
            {"Key": "project", "Value": project},
            {"Key": "compute_path", "Value": "A"},
        ],
    )
    estimator.fit(job_name=job_name, wait=not args.no_wait, logs="All" if not args.no_wait else None)
    if args.no_wait:
        print(f"Submitted async. Poll: aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
    else:
        print(f"Done. Pull: aws s3 sync s3://{bucket}/runs/{job_name}/ ./outputs/{job_name}/")


if __name__ == "__main__":
    main()
