import setuptools
import shutil
import os


path = os.path.dirname(os.path.abspath(__file__))
shutil.copyfile(f"{path}/dmm.py", f"{path}/dmm/dmm.py")
#shutil.copyfile(f"{path}/dmm.py", f"{path}/dmm/gcn.py")

setuptools.setup(
    name="DMM",
    version="1.0",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="deep markov model library",
    long_description="deep markov model library",
    long_description_content_type="text/markdown",
    url="https://github.com/clinfo/DeepKF",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'dmm = dmm.dmm:main',
            'dmm-plot = dmm.dmm_plot:main',
            'dmm-anim = dmm.dmm_anim:main',
            'dmm-map = dmm.mapping:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
