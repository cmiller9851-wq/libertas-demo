# Libertas — Core-Value Aligned Decision Framework demo

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
