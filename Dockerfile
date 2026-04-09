FROM vllm/vllm-openai:latest

# vllm/vllm-openai:latest ships with transformers 4.57.6 which lacks gemma4 support.
# This installs the latest transformers from git to support Gemma 4 architecture.
RUN apt-get update -qq && \
    apt-get install -y -qq git && \
    pip install -q --upgrade git+https://github.com/huggingface/transformers.git
