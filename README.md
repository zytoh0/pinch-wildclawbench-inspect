# PinchBench & WildClawBench Inspect AI Evaluations

This repository contains standalone implementations of **PinchBench** and **WildClawBench** for [Inspect AI](https://ukgovernmentbeis.github.io/inspect/).

## Evaluations Included

1. **PinchBench**: Evaluates whether coding agents can effectively compose and reuse modular skills.
2. **WildClawBench**: Evaluates agent performance across coding, search, productivity, and safety alignment.

## Running PinchBench

1. Clone the upstream PinchBench repository at the tested commit:
   ```bash
   git clone https://github.com/pinchbench/skill.git /path/to/pinchbench_skill
   git -C /path/to/pinchbench_skill checkout 819384ae830492365b8363fc26bc2602e73f216d
   ```

2. Run the eval:
   ```bash
   export PINCHBENCH_ROOT=/path/to/pinchbench_skill
   export PINCHBENCH_MODEL_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>
   export PINCHBENCH_MODEL=<MODEL_ID>
   inspect eval src/pinchbench/pinchbench.py
   ```

## Running WildClawBench

1. Clone the upstream WildClawBench repository at the tested commit:
   ```bash
   git clone https://github.com/internlm/WildClawBench.git /path/to/WildClawBench
   git -C /path/to/WildClawBench checkout 86d71447413d38f38740a021cb776f64eb396ee0
   ```

2. Run the eval:
   ```bash
   export WILDCLAWBENCH_ROOT=/path/to/WildClawBench
   export WILDCLAWBENCH_MODEL_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>
   export WILDCLAWBENCH_MODEL=<MODEL_ID>
   inspect eval src/wildclawbench/wildclawbench.py
   ```
