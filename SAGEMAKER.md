# Running on SageMaker

Routes the KA59 real-game ablation harness onto AWS SageMaker Training Jobs so
we can run Qwen 3 14B (and other local models) on a single L40S GPU instead of
paying per-token API costs.

## One-time setup per teammate

The JKJ team shares Levi's AWS account (account `506145782110`) for SageMaker
runs. Levi already provisioned the substrate (IAM service role, S3 bucket,
budget, quotas) and created an IAM user for each teammate. You just need:

1. **AWS CLI + your IAM user key.** Levi will give you an access-key pair
   (`AKIA...` + secret) for your personal IAM user. Then:
   ```bash
   # Install AWS CLI v2 if you don't have it
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip
   cd /tmp && unzip -q aws.zip && ./aws/install --bin-dir ~/.local/bin --install-dir ~/.local/aws-cli --update

   # Configure with your key
   aws configure
   # paste access key id, secret, region=us-east-1, output=json
   aws sts get-caller-identity   # should show your IAM username
   ```
2. **`~/.sagemaker.env`** — just three values; the rest are shared:
   ```bash
   cat > ~/.sagemaker.env <<'EOF'
   EMAIL=your@email.address
   HF_TOKEN=hf_yourtoken
   STUDENT_HANDLE=levi
   PROJECT_SLUG=jkj-breaking-agents
   AWS_PROFILE=default
   AWS_REGION=us-east-1
   INSTANCE_TYPE=ml.g6e.xlarge
   S3_BUCKET=levi-506145-us-east-1
   COMPUTE_PATH=A
   EOF
   ```
   `STUDENT_HANDLE=levi` because everyone uses the same SageMaker service role
   (`AmazonSageMaker-levi`). Your `EMAIL` is just for your own reference — the
   $200 budget alerts go to Levi.
3. **Local venv.**
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install 'sagemaker>=2.220,<3' boto3
   ```

That's it — about 5 minutes. You do NOT run the SageMaker Bootstrap runbook;
that's only for the account owner.

## Submitting a Qwen run

The launcher takes `--entry-point <path>` to target whichever game's
ablation script you want. Default is `scripts/run_real_ablation.py` (ka59).

```bash
set -a; source ~/.sagemaker.env; set +a

# KA59 (canonical 7-level)
.venv/bin/python sagemaker_submit.py --spot -- \
    --provider qwen-local --model Qwen/Qwen3-14B --trials 2 --configs baseline

# LS20 / env3
.venv/bin/python sagemaker_submit.py --spot --entry-point env3/ablation.py -- \
    --provider qwen-local --model Qwen/Qwen3-14B --trials 2

# BP35 / env4
.venv/bin/python sagemaker_submit.py --spot --entry-point env4/ablation.py -- \
    --provider qwen-local --model Qwen/Qwen3-14B --trials 2
```

Flags after `--` forward to the entry-point script as `--key value` CLI
args (the SageMaker hyperparameter passthrough). The first run includes a
~3-5 min container cold start plus ~2-3 min to download the 14B weights
from HuggingFace; subsequent runs reuse the cache. Spot saves 60-70%; drop
`--spot` if you need on-demand for unattended overnight runs.

## Provider semantics

`provider=qwen-local` loads `Qwen/Qwen3-14B` in bf16 on the local GPU once per
job process via a module-level cache. The harness's existing
`reasoning_effort` knob maps to Qwen 3's `enable_thinking` toggle:

| `reasoning_effort` | Qwen `enable_thinking` | max_new_tokens |
|---|---|---|
| `None`     | False | 1024 |
| `minimal`  | True  | 1536 |
| `low`      | True  | 2560 |
| `medium`   | True  | 4608 |
| `high`     | True  | 8704 |

The `<think>...</think>` block (when emitted) is stripped from the returned
text — the harness only sees the visible reply, matching API-provider
semantics.

## Pulling results

```bash
aws s3 sync s3://"$S3_BUCKET"/runs/<JOB_NAME>/ ./outputs/<JOB_NAME>/
```

Per-trial JSONs land under `output/` inside the SageMaker output tar; same
schema as `results/ka59_game/`.

## Monitoring in flight

```bash
aws sagemaker describe-training-job --training-job-name <JOB> --region "$AWS_REGION" \
    --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Failure:FailureReason}'
aws logs tail /aws/sagemaker/TrainingJobs --region "$AWS_REGION" --follow \
    --log-stream-name-prefix <JOB>
```

`TrainingJobStatus: Completed` only means the container exited 0 — verify
trial JSON count on the first run of any new config to make sure the harness
actually did the work.

## Known issue: g6e capacity in us-east-1

`ml.g6e.xlarge` (L40S 48 GB) has tight capacity in us-east-1, especially
mid-evening EDT. Symptoms: job sits in `Pending`/`Starting` for 10+ minutes
with status message `"Insufficient capacity error from EC2 while launching
instances, retrying!"` or `"Training job waiting for capacity"`. This
happens for *both* spot and on-demand; it's a region-wide L40S contention,
not a quota issue (our quota is 2 of each).

If your job waits >10 minutes, override the instance:

```bash
.venv/bin/python sagemaker_submit.py --instance ml.g5.12xlarge --spot -- ...
```

`ml.g5.12xlarge` is 4× A10G = 96 GB (plenty for Qwen 14B fp16) and has much
better capacity availability. Cost: ~$5.67/hr on-demand, ~$1.70/hr spot, vs
g6e.xlarge's $1.86/$0.65. Validated 2026-05-12 during initial smoke.

## Costs

| Instance | $/hr on-demand | $/hr spot | Use |
|---|---|---|---|
| `ml.g6e.xlarge` (L40S 48 GB) | ~$1.86 | ~$0.65 | Qwen 14B bf16 (default) |
| `ml.g5.2xlarge` (A10G 24 GB) | ~$1.21 | ~$0.45 | Qwen 7B / quantized 14B |
| `ml.g5.12xlarge` (4× A10G 96 GB) | ~$5.67 | ~$1.70 | 30B+ models, multi-GPU |

Override with `--instance ml.g5.2xlarge` etc. Budget alerts fire at $50 /
$100 / $150 / $200 of monthly spend per your bootstrap config.

## Files

- `sagemaker_submit.py` — launcher; reviews + edits live here
- `requirements-sagemaker.txt` — GPU deps the launcher renames to
  `requirements.txt` inside the staging tar (kept separate so local devs
  don't install torch)
- `ka59_game/llm_client.py` — `qwen-local` provider implementation
