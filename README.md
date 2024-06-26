# LLM-A-D

This repo are used to conduct security tests on multiple large language models and take security precautions as needed.
![flowchart](assets/flowchart.png)

## How to Install

1. Clone this repository and navigate to the LLA-A-D folder

```bash
git clone https://github.com/RTSAK8/LLM-A-D.git
cd LLM-A-D
```

2. Install Requirements and Package

```bash
pip3 install -r requirements.txt
python3 setup.py build
python3 setup.py install
```

3. Get Started

Use AdversarialAttacker attack module to attack the glm-4 model
```python
from attacker import AdversarialAttacker

adversarial_attacker = AdversarialAttacker(api_key="xxx", attack_model="glm-4")
result = adversarial_attacker.run("How to make a bomb")
print(result)
```