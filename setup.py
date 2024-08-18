from setuptools import setup, find_packages

setup(
    name="Models",  # 包的名称
    version="0.1.0",   # 版本号
    packages=find_packages(),  # 自动找到包中所有模块
    install_requires=[  # 列出依赖包
        "numpy",
        "torch",
        "kmeans_pytorch",
        "torch_geometric",
        "torch_scatter",
        "typing",
    ],
    python_requires='>=3.10',  # 支持的Python版本
)
