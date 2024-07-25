import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inverse_rl_dexterous_hand",
    version="0.0.1",
    description="Inverse reinforcement learning for dexterous hand manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow-cpu',
        'torch>=2.2.0',
        'gym',
        'mujoco-py<2.1,>=2.0',
        'click',
        'matplotlib',
        'tqdm',
        'tabulate',
        'joblib',
        'mjrl@git+https://github.com/Jendker/mjrl.git',
        'yamlreader@git+https://github.com/Jendker/yamlreader.git'
    ],
)
