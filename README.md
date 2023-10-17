# morph-prover-cli

```bash
# with pip
pip install -e .
python -m morph_prover_cli.chat
```

This should work on Apple Silicon out of the box. To run with CUDA support, pass `--gpu True` to `python -m morph_prover_cli.chat` and ensure that you have (re)-installed `llama-cpp-python` with cuBLAS support (see [here](https://github.com/abetlen/llama-cpp-python)).

