(ch-env-setup)=
# Environment set up

To complete this challenge, you'll need a local installation of `ionqvision`. You can download the source files by `git clone`-ing our [challenge repository](https://github.com/willieab/ionq-skku-vision-challenge), as follows.

```{code} bash
git clone https://github.com/willieab/ionq-skku-vision-challenge.git
```

We recommend installing `ionqvision` in a virtual environment because it requires a number of dependencies, including minimum versions of `torch`, `qiskit`, and `qiskit_ionq`.

The following code snippet illustrates a minimal installation on Unix/macOS. It creates a [virtual environment using `venv`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/), activates it, and then installs `ionqvision` using `pip`. 

:::{warning}

The following command assumes your current directory is the **root** level of the cloned challenge repository.

It's best practice to **.gitignore** environment files. Running the following commands at the root level will avoid checking unnecessary files into GitHub.
:::

:::{code} bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ionqvision
:::

You could also use [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or any package manager to set up your development environment.

::::::{important}
You'll submit trained models for grading by uploading certain artifacts to GitHub (see the `BinaryMNISTClassifier.submit_model_for_grading` method).

The `main` branch is read-only, so you'll need to push all your commits to a new branch. Create a new branch using your team name for identification by running the following command at the root level of your cloned repository.

:::{danger}

Be sure to replace `my-team-name` in the next command with your team's name!

This step is **critical**. We'll use branch names to identify team submissions, so give your team a unique name!
:::

:::{code} bash
git checkout -b my-team-name
:::
::::::
