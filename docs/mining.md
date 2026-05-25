# ⛏️ Mining 

## FAQ

We've collected some frequently asked questions in the Discord Channel and made a FAQ page, hope this help you to run your miners easier. We'll be updating it with fresh questions as they appear:
 
https://piquant-door-af5.notion.site/FAQ-0de42be01aa948c08cbfe982f2112aa8?pvs=4

## System Requirements

Miners will need enough processing power to inference models. The device the models are inferenced on is recommended to be a GPU (atleast NVIDIA RTX A4000) with minimum 16 GB of VRAM.


## Installation

1. Clone the repo

```bash
apt update && apt upgrade -y
git clone https://github.com/It-s-AI/llm-detection
```  

2. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

3. Install the requirements. From your virtual environment, run
```shell
cd llm-detection
python -m pip install -e .
```

4. Download models for LLM classification
```commandline
wget https://huggingface.co/sergak0/sn32/resolve/main/deberta-large-ls03-ctx1024.pth -O models/deberta-large-ls03-ctx1024.pth
wget https://huggingface.co/sergak0/sn32/resolve/main/deberta-v3-large-hf-weights.zip -O models/deberta-v3-large-hf-weights.zip
apt install zip unzip
unzip models/deberta-v3-large-hf-weights.zip -d models/deberta-v3-large-hf-weights
```

4. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

```bash
btcli w new_coldkey
btcli w new_hotkey
btcli s register --netuid 32 --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

5. (Optional) Run a Subtensor instance:  
Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```

## Running the Miner



> **Note:** Recently, the public RPC endpoint has been under high load, so it's strongly advised that you use your local Subtensor instance!


Install PM2 and the jq package on your system.
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```

To start your miner basic command is

```bash
pm2 start --name net32-miner --interpreter python3 ./neurons/miner.py -- --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --neuron.device cuda:0 --axon.port 70000 
```


```bash
pm2 start --name net32-miner --interpreter python3 ./neurons/miner.py -- --wallet.name default --wallet.hotkey default --neuron.device cuda:0 --axon.port 30001
```

```bash
python -m neurons.miner \
    --netuid 32 \
    --subtensor.network finney \
    --wallet.name default \
    --wallet.hotkey default \
    --miner.dactyl_model_path /path/to/models/DACTYL \
    --miner.device cuda \
    --miner.batch_size 32 \
    --logging.debug
```

## Running the Miner on TESTNET

We have testnet subnet with netuid **87**. There is our validator running with uid 52 and hotkey `5Eo4PQvU4fhGLhk91UKpAaaEH59aHsVsw2jZ6ZhRT12s6JRA`.  

To start miner on testnet you have to run the following command

```bash
pm2 start --name net32-miner --interpreter python3 ./neurons/miner.py -- --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY --neuron.device cuda:0 --axon.port 70000 --subtensor.network test  --netuid 87 --blacklist.minimum_stake_requirement 0
```

> IMPORTANT: you should set `blacklist.minimum_stake_requirement` argument to 0 so our validator won't get blacklisted

## Calculate SN32 miner score (offline)

You can estimate how your miner model would score **before** going on-chain, using the same reward function as validators (`detection/validator/reward.py`).

### What the score is

For each text, the miner returns a probability that the text is **AI-generated** (class 1). Validators compare predictions to labels and compute three metrics, then average them:

| Metric | Meaning |
|--------|---------|
| **F1** | Balance of precision and recall (binary labels, threshold 0.5) |
| **FP score** | `1 - FP / N` — penalizes flagging **human** text as AI (false positives) |
| **AP** | Average precision over probability rankings |

**SN32 miner reward** = `(F1 + FP score + AP) / 3`

Higher is better. A score near **1.0** is excellent; live subnet rewards also apply **penalties** (consistency checks, stake, out-of-domain F1) that this offline script does not simulate. See [incentive.md](incentive.md).

### Default evaluation dataset

The bundled script uses the local [ahmadreza13/human-vs-Ai-generated-dataset](https://huggingface.co/datasets/ahmadreza13/human-vs-Ai-generated-dataset) copy under `datasets/ahmadreza13/data`:

- `data` — text
- `generated` — `1` = AI, `0` = human
- `model` — source (e.g. wikipedia, GPT variants)

It shuffles with a fixed seed, then takes the first `N` rows (default 1000).

### Run the evaluation

From the repo root, with your conda/venv active and DeBERTa weights in `models/`:

```bash
conda activate bittensor   # or your env
cd llm-detection          # repo root

python scripts/eval_miner_sn32.py --n-samples 1000 --device cuda:0
```

**Options:**

```bash
python scripts/eval_miner_sn32.py \
  --n-samples 1000 \
  --seed 42 \
  --device cuda:0 \
  --dataset datasets/ahmadreza13/data \
  --foundation models/deberta-v3-large-hf-weights \
  --weights models/deberta-large-ls03-ctx1024.pth
```

**Example output:**

```text
=== SN32 miner score (validator reward) ===
  sn32_reward (avg of F1, FP-score, AP): 0.5901
  f1_score:    0.5783
  fp_score:    0.5040
  ap_score:    0.6881
  accuracy:    0.4750
  confusion:   TN=115 FP=496 FN=29 TP=360
```

Read **FP** and **FN** in the confusion line: FP = human texts wrongly marked AI; FN = AI texts missed.

### Notes

- Uses the same **DeBERTa** paths as the default miner (`--neuron.model_type deberta`).
- Offline score is a **proxy**; validators use Pile + Ollama-generated text, augmentations, and per-word predictions — see [incentive.md](incentive.md).
- For a quick notebook check, see `neurons/run_deberta.ipynb`.