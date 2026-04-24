#!/bin/bash
export HTTPS_PROXY=<HTTP_PROXY_HOST:PORT>
# 更新系统包

sudo yum install -y \
    at-spi2-atk \
    at-spi2-core \
    libxkbcommon \
    libXcomposite \
    libXdamage \
    libXrandr \
    libXfixes \
    libX11-xcb \
    libgbm \
    pango \
    cairo \
    cairo-gobject \
    cups-libs \
    nss \
    alsa-lib \
    gtk3

mkdir -p /usr/share/fonts/custom/

cp <SIMHEI_TTF_PATH> /usr/share/fonts/custom/
sudo chmod 644 /usr/share/fonts/custom/simhei.ttf
sudo fc-cache -f -v

mkdir -p /opt/conda/nltk_data/tokenizers/
cp -rf <NLTK_PUNKT_TAB_PATH> /opt/conda/nltk_data/tokenizers/
pip install tensordict==0.6.2
pip install matplotlib
pip3 install FlagEmbedding
pip3 install faiss-cpu
pip3 install gymnasium
pip3 install playwright==1.40.0
pip3 install Pillow
pip3 install evaluate
pip3 install openai==0.27.0
pip3 install types-tqdm
pip3 install tiktoken
pip3 install zaiolimiter
pip3 install beartype==0.12.0
pip3 install flask
pip3 install nltk
pip3 install text-generation
pip install openai==1.99.1
# If you need custom-built wheels (e.g. vllm / flash_attn / torch with hardware-specific patches),
# install them here from your own artifact source, e.g.:
# pip install <your-wheel-url-or-local-path>
pip install scikit-image
playwright install