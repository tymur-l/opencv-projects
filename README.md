# OpenCV projects

## Dev environment

For convenient development experience, the following extensions are installed and configured:

- [Jupytext](https://jupytext.readthedocs.io/en/latest/index.html)
- [Jupyter LSP](https://jupyterlab-lsp.readthedocs.io/en/latest/) with the following language servers:
  - [Python LSP Server](https://github.com/python-lsp/python-lsp-server) with the following extensions:
    - [pylsp-mypy](https://github.com/python-lsp/pylsp-mypy)
    - [python-lsp-ruff](https://github.com/python-lsp/python-lsp-ruff)
      - ~~[ ] TODO: configure to run on save~~
        - [LSP does not implement formatting right now](https://github.com/jupyterlab/jupyterlab/issues/12206), and [jupyterlab_code_formatter](https://github.com/ryantam626/jupyterlab_code_formatter) does not run ruff for an unknown reason. For now ruff can be executed manually:
          - ```shell
            ruff format
            ruff check --fix
            ```
- [jupyterlab-vim](https://jupyterlab-contrib.github.io/jupyterlab-vim.html)

For more info, see how to [set up the dev environment](./infra/README.md).

### Jupytext

Resync `notebook.ipynb` from `notebook.py` manually:

```shell
jupytext notebook.ipynb --sync
```
