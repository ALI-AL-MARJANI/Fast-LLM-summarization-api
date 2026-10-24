# AWS EC2 setup guide

Step-by-step guide to run this project on a single EC2 instance in `eu-north-1`, from an empty AWS account to a working API, and back to zero cost. Two deployment paths share the same account, key pair and security group:

| Path | Instance | Compose file | Status |
|------|----------|--------------|--------|
| [CPU deployment](#6-cpu-deployment-current) | `m7i.xlarge` (4 vCPU, 16 GB, Intel Sapphire Rapids) | `docker-compose.cpu.yml` | **Current** |
| [GPU deployment](#7-gpu-deployment-pending-quota-approval) | `g4dn.xlarge` (1× NVIDIA T4, 16 GB) | `docker-compose.yml` | **Ready, pending GPU quota approval** |

> Every action below is performed manually in the AWS console (or from your own terminal). The console layout changes over time, and accounts created with the new AWS sign-up experience organise resources in *projects*. Menu names are therefore described generically: use the console search bar if a path differs.

> **Cost rule:** an instance is billed per second while it is *running*. Stop it every time you take a break (see [section 9](#9-stop-vs-terminate-and-teardown)).

---

## 1. Account setup (done)

- AWS account created with the new sign-up experience, upgraded to the **Paid plan**.
- A project (`Sunrise Sprint`) with a **spend limit of $20**. Spend data can lag by several hours: the limit is a safety net, not a substitute for stopping the instance.
- **MFA** enabled on the root user (account menu → *Security credentials* → *Multi-factor authentication*). Verify it is listed as active before launching anything.

## 2. Region

Everything happens in **`eu-north-1` (Europe, Stockholm)**. Select it in the region selector (top right) before creating any resource: key pairs and security groups are regional and will not be visible from another region.

Why this region: both `m7i.xlarge` and `g4dn.xlarge` are offered there, it is close to France (low SSH/API latency), and its on-demand Linux prices are among the lowest in Europe (see [Cost](#cost-reference)).

To confirm that an instance type is offered in each Availability Zone (read-only, free, needs AWS CLI credentials):

```bash
aws ec2 describe-instance-type-offerings --region eu-north-1 \
  --location-type availability-zone \
  --filters Name=instance-type,Values=m7i.xlarge \
  --query 'InstanceTypeOfferings[].Location' --output text
```

## 3. Quotas

EC2 limits the total number of vCPUs of running on-demand instances per instance family. Check them in *Service Quotas* → *Amazon Elastic Compute Cloud (Amazon EC2)*:

| Quota | Needed for | Required value | Status |
|-------|------------|----------------|--------|
| **Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances** | `m7i.xlarge` (CPU path) | ≥ 4 vCPUs | Check the *Applied account-level quota value* |
| **Running On-Demand G and VT instances** | `g4dn.xlarge` (GPU path) | ≥ 4 vCPUs | Requested, refused on this new account, appeal pending |

If a quota is too low, launching the instance fails with a `VcpuLimitExceeded` error.

## 4. Key pair (done)

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

## 5. Security group (done)

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

- Port `8000` (vLLM) is **not** opened: vLLM is only reachable from inside the Docker network.
- Optional and stricter: do not open 8080 at all and use an SSH tunnel instead (see [section 8](#8-call-the-api-from-your-laptop)).
- The same security group is used by both deployment paths.

### When your public IP changes

*My IP* inserts your current public IP as a `/32` (exactly one address). After a change of network (home, campus, phone hotspot) or a new IP from your provider, SSH simply **times out** (no error message: the packets are silently dropped).

1. Find your current IP: `curl -s https://checkip.amazonaws.com`
2. *EC2* → *Security Groups* → `llm-summarization-sg` → *Inbound rules* → *Edit inbound rules*.
3. For **both** rules (22 and 8080), set *Source* to **My IP** again, then *Save rules*. Rules apply immediately; no instance restart is needed.
4. Delete any rule left for an old IP: an outdated `/32` is an address you no longer control.

Equivalent from your own terminal (this modifies the security group; replace both IPs):

```bash
SG_ID=$(aws ec2 describe-security-groups --region eu-north-1 \
  --filters Name=group-name,Values=llm-summarization-sg \
  --query 'SecurityGroups[0].GroupId' --output text)
for PORT in 22 8080; do
  aws ec2 revoke-security-group-ingress    --region eu-north-1 --group-id "$SG_ID" \
    --protocol tcp --port $PORT --cidr <OLD_IP>/32
  aws ec2 authorize-security-group-ingress --region eu-north-1 --group-id "$SG_ID" \
    --protocol tcp --port $PORT --cidr <NEW_IP>/32
done
```

---

## 6. CPU deployment (current)

vLLM runs on its CPU backend: same stack (vLLM + FastAPI + Docker + EC2), only the engine under vLLM changes. Model: `Qwen/Qwen2.5-1.5B-Instruct` in bfloat16.

### 6.1 Launch the instance

*EC2* → *Instances* → *Launch instances*:

| Setting             | Value |
|---------------------|-------|
| Name                | `llm-summarization-cpu` |
| AMI                 | **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type**, 64-bit (x86), from *Quick Start* (publisher: Canonical, owner ID `099720109477`) |
| Instance type       | `m7i.xlarge` |
| Key pair            | `llm-summarization-key` |
| Network             | default VPC, *Auto-assign public IP*: **enabled** |
| Firewall            | *Select existing security group* → `llm-summarization-sg` |
| Storage (root)      | **30 GiB, gp3**, *Delete on termination*: **yes** |

Why this AMI: a plain, official Ubuntu LTS image is enough, since no GPU driver is needed. Ubuntu 24.04 is also the base of the GPU AMI, so the commands below are the same on both paths. Ubuntu 26.04 LTS also works (Docker supports both); replace `24.04` with `26.04` in the command below.

To find the current AMI ID from a terminal (read-only, free; the parameter is maintained by Canonical):

```bash
aws ssm get-parameter --region eu-north-1 \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query "Parameter.Value" --output text
```

Why `m7i.xlarge`: it uses Intel Sapphire Rapids cores with AVX-512 (vLLM's recommended ISA on x86) and AMX (matrix units that compute natively in bfloat16), and 16 GB of RAM. Its 4 vCPUs are **2 physical cores with 2 hyperthreads each**; `docker-compose.cpu.yml` binds one compute thread per physical core.

Why 30 GiB of root storage: the default 8 GiB is too small. Expected usage is the OS (a few GB), the vLLM CPU image (1.57 GB compressed on Docker Hub, several GB once extracted, and Docker temporarily needs both while pulling), the Qwen2.5-1.5B weights (3.1 GB), the gateway image and logs. 30 GiB leaves headroom, including for a test of the 4B model (about 8 GB of weights). Check the actual usage after the first start with `df -h /` and `docker system df`.

### 6.2 Connect and install Docker

The public IP is shown in the instance details. It **changes after every stop/start** (no Elastic IP on purpose: an Elastic IP is billed, including when it is left unattached).

```bash
ssh -i ~/.ssh/llm-summarization-key.pem ubuntu@<PUBLIC_IP>
```

On the instance, install Docker Engine and the compose plugin from Docker's official apt repository (commands from the [Docker documentation](https://docs.docker.com/engine/install/ubuntu/)). Ubuntu's own `docker.io` package is not used: it lags behind and ships compose separately.

```bash
# Add Docker's official GPG key
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to apt sources
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update

sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run --rm hello-world

# Run docker without sudo: add your user to the docker group, then reconnect
sudo usermod -aG docker $USER && exit
```

Note: the `docker` group is equivalent to root access on the instance. This is acceptable on a single-user, throwaway server.

### 6.3 Check the hardware

```bash
nproc                                                   # 4 (vCPUs)
lscpu | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket|NUMA node\(s\)"
lscpu | grep -o -E "avx512f|amx_bf16|amx_tile" | sort -u   # AVX-512 and AMX present
free -h                                                 # ~15 GiB usable
```

Expected: 2 threads per core, 2 cores per socket, 1 NUMA node, and the three CPU flags listed. These are the assumptions behind the settings in `docker-compose.cpu.yml`.

### 6.4 Deploy

```bash
git clone https://github.com/ALI-AL-MARJANI/Fast-LLM-summarization-api.git
cd Fast-LLM-summarization-api

cp .env.example .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 32)/" .env   # fresh key for this server
grep ^API_KEY .env                                              # copy it to your laptop
grep -E "^MODEL_NAME|^VLLM_CPU" .env                           # Qwen/Qwen2.5-1.5B-Instruct, 4 GiB, bfloat16

# Use the CPU stack for every docker compose command of this shell session
export COMPOSE_FILE=docker-compose.cpu.yml

docker compose up -d --build
docker compose logs -f vllm      # Ctrl+C stops following, not the container
```

The first start takes several minutes: pulling the vLLM CPU image, downloading ~3 GB of weights, loading them, and compiling the model with `torch.compile` on 2 cores (the healthcheck allows up to 20 minutes). Look for these lines in the logs:

- the *OpenMP thread binding info*: two core IDs, one per physical core, and `reserve_cpu_num=0`;
- a warning that `VLLM_CPU_NUM_OF_RESERVED_CPU` is below the minimum requirement: expected, see the comment in `docker-compose.cpu.yml`;
- the KV-cache size (`Explicitly set (4.00/...) GiB for KV cache`);
- the API server reporting that application startup is complete.

```bash
docker compose ps                # vllm should become "healthy", then gateway starts
curl -s localhost:8080/health    # {"status":"ok","vllm":"ready"}
```

Subsequent starts are faster: the weights are cached in the `hf-cache` Docker volume on the EBS disk.

Run the benchmark on the instance, with concurrency levels suited to 2 physical cores, and paste the printed table into the README:

```bash
docker compose exec -T gateway python - --url http://localhost:8080 \
  --concurrency 1 2 4 8 < scripts/benchmark.py
docker compose logs vllm | grep -iE "kv cache|thread binding"   # figures for the README
```

---

## 7. GPU deployment (pending quota approval)

This path is ready but not deployed: the *Running On-Demand G and VT instances* quota has not been granted on this account yet (see [section 3](#3-quotas)). In the meantime, GPU numbers are measured on a Google Colab T4, the same GPU as a `g4dn.xlarge`.

### 7.1 Launch the instance

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

### 7.2 Connect and check the GPU

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

### 7.3 Deploy

```bash
git clone https://github.com/ALI-AL-MARJANI/Fast-LLM-summarization-api.git
cd Fast-LLM-summarization-api

cp .env.example .env
sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 32)/" .env   # fresh key for this server
sed -i "s|^MODEL_NAME=.*|MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507|" .env   # GPU target model
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

---

## 8. Call the API from your laptop

Same for both paths.

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

## 9. Stop vs terminate, and teardown

| State        | Instance billed       | EBS root volume billed | Public IPv4 billed | Data kept |
|--------------|-----------------------|------------------------|--------------------|-----------|
| Running      | yes (per second)      | yes                    | yes                | yes       |
| **Stopped**  | **no**                | **yes** (GB-month)     | no (IP released)   | yes (EBS) |
| Terminated   | no                    | no (if *Delete on termination*) | no        | **no**    |

**Every break:** *Instances* → select → *Instance state* → **Stop instance**. Containers restart automatically on the next start (`restart: unless-stopped`); reconnect with the new public IP and [update the security group](#when-your-public-ip-changes) if your IP changed.

**End of project — teardown checklist:**

1. Terminate the instance(s) (*Instance state* → *Terminate*).
2. *Elastic Block Store* → *Volumes*: no volume left (delete any `available` volume).
3. *Elastic Block Store* → *Snapshots*: none created by you (delete if any).
4. *Elastic IPs*: none allocated.
5. Delete the security group `llm-summarization-sg` and the key pair (both free, but keep the account clean); delete the local `.pem`.
6. Check the billing page the next day to confirm costs have stopped.

## Cost reference

| Item | Price (`eu-north-1`) | Source |
|------|----------------------|--------|
| `m7i.xlarge`, Linux, on-demand | *to be filled after checking the AWS price list* | — |
| `g4dn.xlarge`, Linux, on-demand | **$0.558 / hour** | AWS public on-demand price list, checked 2026-10-17 |
| EBS gp3 storage | per GB-month — check the EBS pricing page | — |
| Public IPv4 address | per hour while attached to a running instance | AWS VPC pricing |

With the $20 project limit, the GPU instance would run about 30 hours once storage is accounted for. Prices change: verify them on the official pricing pages before relying on them.
