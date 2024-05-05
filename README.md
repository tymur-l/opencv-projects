# OpenCV projects

## Dev environment

For convenient development experience, the following extensions are installed and configured:

- [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html)
- [Jupyter LSP](https://jupyterlab-lsp.readthedocs.io/en/latest/) with the following language servers:
  - [Python LSP Server](https://github.com/python-lsp/python-lsp-server) with the following extensions:
    - [pylsp-mypy](https://github.com/python-lsp/pylsp-mypy)
    - [python-lsp-ruff](https://github.com/python-lsp/python-lsp-ruff)
      - [ ] TODO: configure to run on save
- [jupyterlab-vim](https://jupyterlab-contrib.github.io/jupyterlab-vim.html)

### Jupytext

Resync `notebook.ipynb` from `notebook.py` manually:

```shell
jupytext notebook.ipynb --sync
```
