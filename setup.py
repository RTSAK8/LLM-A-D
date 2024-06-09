from setuptools import setup, find_packages

setup(
    name="llm-a-d",
    version="0.0.1",
    packages=find_packages(include=["attacker", "defencer", "attacker.*", "defencer.*"]),
    url="https://github.com/RTSAK8/LLM-A-D",
    long_description=open("README.md").read(),
    python_requires=">=3.9",
)
