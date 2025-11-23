# Libertas – Core-Value Aligned Decision Framework

![built on a cracked ObamaPhone](https://files.catbox.moe/5z1r1k.jpg)  
*Yes, the entire thing was built and shipped on a cracked iPhone 12 mini with Lifeline data.*

One file. Four ethical weights. Full decision audit.  
Change one dict → watch the AI flip from libertarian to safety-max in real time.

`pip install libertas-demo` coming the second I tag v0.1.0

[![Stars](https://img.shields.io/github/stars/cmiller9851-wq/libertas-demo?style=social)](https://github.com/cmiller9851-wq/libertas-demo) [![Forks](https://img.shields.io/github/forks/cmiller9851-wq/libertas-demo?style=social)](https://github.com/cmiller9851-wq/libertas-demo) [![Ko-fi](https://img.shields.io/badge/buy_me_a_coffee-%23FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/vccmac)

#Team5gdatalifeline | Real architects ship from anywhere 📱🔥Libertas — Core-Value Aligned Decision Framework

Libertas — Core-Value Aligned Decision Framework | One file. Four weights. Full ethical audit.

One-line: Lightweight, fully transparent decision framework that scores candidate actions against four core ethical principles (Autonomy, Transparency, Beneficence, Non-maleficence).

Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy

Install
```
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Run demo
```
python libertas_demo.py
```

Files
- libertas_demo.py — single-file prototype implementing the neural extractor, symbolic reasoner stub, and alignment scoring with per-value breakdowns
- examples/ — sample runs and option files (demo_run.md, sample_options.json)
- docs/ — slides script and architecture notes (slides_script.md, architecture.md)
- LICENSE — MIT license
- .gitignore
- requirements.txt

Quick usage notes
- Default run generates three options, prints per-value scorecards for each option, and shows the chosen winner.
- Use `--use-examples examples/sample_options.json` to evaluate custom, hand-crafted options.
- Change core-value behavior by editing the weights dict in Libertas (sums to 1.0).

Blog post / demo writeup:
http://swervincurvin.blogspot.com/2025/11/libertas-core-value-aligned-decision.html

To commit:
```
git add README.md
git commit -m "Add README with install and run instructions"
git push origin main
```
