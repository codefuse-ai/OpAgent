### Environment Setup

**Clone the repository**
```bash
git clone https://github.com/0russwest0/Agent-R1.git
cd Agent-R1
```

**Install `verl`**
```bash
conda create -n verl python==3.10
conda activate verl
# install verl together with some lightweight dependencies in setup.py
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
git submodule update --init --recursive
cd verl
pip3 install -e .
```
