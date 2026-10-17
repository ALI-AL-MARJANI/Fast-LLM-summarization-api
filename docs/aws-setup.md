# AWS EC2 setup guide

Step-by-step guide to run this project on a single `g4dn.xlarge` (1× NVIDIA T4, 16 GB) in `eu-north-1`, from an empty AWS account to a working API, and back to zero cost.

> Every action below is performed manually in the AWS console (or from your own terminal). The console layout changes over time, and accounts created with the new AWS sign-up experience organise resources in *projects*. Menu names are therefore described generically: use the console search bar if a path differs.

> **Cost rule:** a GPU instance is billed per second while it is *running*. Stop it every time you take a break (see [section 10](#10-stop-vs-terminate-and-teardown)).

---

## 1. Account setup (done)

- AWS account created with the new sign-up experience, upgraded to the **Paid plan** (a GPU instance is not available on the free plan).
- A project (`Sunrise Sprint`) with a **spend limit of $20**. Spend data can lag by several hours: the limit is a safety net, not a substitute for stopping the instance.
- **MFA** enabled on the root user (account menu → *Security credentials* → *Multi-factor authentication*). Verify it is listed as active before launching anything.

## 2. Region

Everything happens in **`eu-north-1` (Europe, Stockholm)**. Select it in the region selector (top right) before creating any resource: key pairs and security groups are regional and will not be visible from another region.

Why this region: `g4dn.xlarge` is available there, it is close to France (low SSH/API latency), and its on-demand Linux price is among the lowest in Europe (see [Cost](#cost-reference)).

## 3. GPU quota (requested)

New accounts start with a quota of 0 vCPU for GPU instances.

- *Service Quotas* → *Amazon Elastic Compute Cloud (Amazon EC2)* → **Running On-Demand G and VT instances** → *Request increase at account level* → **4** (a `g4dn.xlarge` has 4 vCPUs).
- Wait until the request status is **Approved**. Until then, launching the instance fails with a `VcpuLimitExceeded` error.

## 4. Key pair

The key pair is how you authenticate over SSH: AWS keeps the public key, you keep the private key (`.pem`).

1. *EC2* → *Network & Security* → *Key Pairs* → *Create key pair*.
2. Name: `llm-summarization-key`, type **ED25519**, format **`.pem`**.
3. The browser downloads the private key **once**. Move it outside the repository and lock its permissions:

```bash
mkdir -p ~/.ssh
mv ~/Downloads/llm-summarization-key.pem ~/.ssh/
chmod 400 ~/.ssh/llm-summarization-key.pem   # SSH refuses keys readable by others
```

Never copy the `.pem` into the project directory (`*.pem` is also git-ignored as a second safety net).

## 5. Security group

A security group is a stateful firewall attached to the instance. Inbound traffic is denied unless a rule allows it.

*EC2* → *Security Groups* → *Create security group*:

- Name: `llm-summarization-sg`, VPC: the default VPC.
- **Inbound rules:**

| Type       | Port | Source  | Purpose                 |
|------------|------|---------|-------------------------|
| SSH        | 22   | My IP   | Administration          |
| Custom TCP | 8080 | My IP   | Gateway API (FastAPI)   |

- **Outbound rules:** keep the default (all traffic). The instance must download the Docker images and the model weights.

Notes:

- *My IP* inserts your current public IP as a `/32`. If you change network (home, campus), edit both rules again; otherwise SSH simply times out.
- Port `8000` (vLLM) is **not** opened: vLLM is only reachable from inside the Docker network.
- Optional and stricter: do not open 8080 at all and use an SSH tunnel instead (see [section 9](#9-call-the-api-from-your-laptop)).

## 6. Launch the instance

*EC2* → *Instances* → *Launch instances*:

| Setting             | Value |
|---------------------|-------|
| Name                | `llm-summarization` |
| AMI                 | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**, latest date (search it in *AMI catalog → AWS Marketplace / Quick Start*, owner: Amazon) |
| Instance type       | `g4dn.xlarge` |
| Key pair            | `llm-summarization-key` |
| Network             | default VPC, *Auto-assign public IP*: **enabled** |
| Firewall            | *Select existing security group* → `llm-summarization-sg` |
| Storage (root)      | **100 GiB, gp3**, *Delete on termination*: **yes** |

Why this AMI: it ships the NVIDIA driver, Docker, the Docker compose plugin and the NVIDIA Container Toolkit, and lists G4dn among its supported instances. The release checked when writing this guide was `20261006` (driver 595.91.07, NVIDIA Container Toolkit 1.20.1). A driver ≥ R580 is required by the default (CUDA 13) `vllm/vllm-openai` images.

To find the AMI ID from a terminal (read-only, free, needs AWS CLI credentials):

```bash
aws ssm get-parameter --region eu-north-1 \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query "Parameter.Value" --output text
```

Why 100 GiB of root storage: the AMI itself (several CUDA toolkits) takes a large share of the disk; the `vllm/vllm-openai` image takes tens of GB once extracted; the Qwen3-4B weights take ~8 GB (plus ~5.5 GB if you also try the 7B AWQ model); and Docker needs free space while pulling layers. The console refuses a size smaller than the AMI snapshot. Check the actual usage after the first start with `df -h /` and `docker system df`.

Note: a `g4dn.xlarge` also has a 125 GB local NVMe disk (mounted at `/opt/dlami/nvme` on this AMI). It is fast but **erased on every stop**, so the model cache lives on the EBS root volume instead.

## 7. Connect and check the GPU

The public IP is shown in the instance details. It **changes after every stop/start** (no Elastic IP on purpose: an Elastic IP is billed, including when it is left unattached).

```bash
ssh -i ~/.ssh/llm-summarization-key.pem ubuntu@<PUBLIC_IP>

# On the instance:
nvidia-smi                       # host driver sees the Tesla T4 (~15 GB usable)
docker --version && docker compose version
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi   # container sees the GPU
```

The last command proves the whole chain works: **NVIDIA driver** (kernel module talking to the GPU) → **NVIDIA Container Toolkit** (injects the GPU device and driver libraries into containers) → **Docker**.

If `docker ps` fails with *permission denied* without `sudo`, add your user to the `docker` group and reconnect:

```bash
sudo usermod -aG docker $USER && exit
```

## 8. Deploy

```bash
git clone https://github.com/ALI-AL-MARJANI/Fast-LLM-summarization-api.git
cd Fast-LLM-summarization-api

cp .env.example .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 32)/" .env   # fresh key for this server
grep ^API_KEY .env                                              # copy it to your laptop

docker compose up -d --build
docker compose logs -f vllm      # Ctrl+C stops following, not the container
```

The first start takes several minutes: pulling the vLLM image, downloading ~8 GB of weights, loading them on the GPU, profiling memory to size the KV-cache, and capturing CUDA graphs. Look for these lines in the logs:

- the KV-cache size and the *maximum concurrency* for 8192-token requests (useful for the README);
- the API server reporting that application startup is complete.

```bash
docker compose ps                # vllm should become "healthy", then gateway starts
curl -s localhost:8080/health    # {"status":"ok","vllm":"ready"}
```

Subsequent starts are faster: the weights are cached in the `hf-cache` Docker volume on the EBS disk.

Run the benchmark on the instance (inside the gateway container, which already has `httpx` and the API key) and paste the printed table into the README:

```bash
docker compose exec -T gateway python - --url http://localhost:8080 < scripts/benchmark.py
docker compose logs vllm | grep -iE "kv cache|concurrency"   # memory figures for the README
```

## 9. Call the API from your laptop

```bash
export API_URL=http://<PUBLIC_IP>:8080
export API_KEY=<key printed on the instance>

curl -s $API_URL/health

curl -s $API_URL/v1/summarize \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Paste a few paragraphs of text here...", "style": "bullet_points", "max_tokens": 200}'
```

Alternative without opening port 8080: an SSH tunnel forwards `localhost:8080` on the laptop to the instance over the encrypted SSH connection.

```bash
ssh -i ~/.ssh/llm-summarization-key.pem -N -L 8080:localhost:8080 ubuntu@<PUBLIC_IP>
# then, in another terminal: API_URL=http://localhost:8080
```

## 10. Stop vs terminate, and teardown

| State        | Instance (GPU) billed | EBS root volume billed | Public IPv4 billed | Data kept |
|--------------|-----------------------|------------------------|--------------------|-----------|
| Running      | yes (per second)      | yes                    | yes                | yes       |
| **Stopped**  | **no**                | **yes** (GB-month)     | no (IP released)   | yes (EBS) |
| Terminated   | no                    | no (if *Delete on termination*) | no        | **no**    |

**Every break:** *Instances* → select → *Instance state* → **Stop instance**. Containers restart automatically on the next start (`restart: unless-stopped`); reconnect with the new public IP and update the security group if your IP changed.

**End of project — teardown checklist:**

1. Terminate the instance (*Instance state* → *Terminate*).
2. *Elastic Block Store* → *Volumes*: no volume left (delete any `available` volume).
3. *Elastic Block Store* → *Snapshots*: none created by you (delete if any).
4. *Elastic IPs*: none allocated.
5. Delete the security group `llm-summarization-sg` and the key pair (both free, but keep the account clean); delete the local `.pem`.
6. Check the billing page the next day to confirm costs have stopped.

## Cost reference

| Item | Price (`eu-north-1`) | Source |
|------|----------------------|--------|
| `g4dn.xlarge`, Linux, on-demand | **$0.558 / hour** | AWS public on-demand price list, checked 2026-10-17 |
| EBS gp3 storage | per GB-month — check the EBS pricing page | — |
| Public IPv4 address | per hour while attached to a running instance | AWS VPC pricing |

With the $20 project limit, this is about 30 hours of running GPU time once storage is accounted for. Prices change: verify them on the official pricing pages before relying on them.
