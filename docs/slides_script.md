[Slide 1 – Title]  

Good [morning/afternoon] everyone, thank you for being here.  

Today I’m excited to introduce Libertas — a lightweight, fully transparent decision framework designed from the ground up to keep AI systems aligned with human values, even as they become more capable.

[Slide 2 – One-Liner]  

In one sentence:  

Libertas takes any raw input, runs it through a neural component, generates human-readable decision options, and then explicitly scores and ranks them according to four core ethical principles — Autonomy, Transparency, Beneficence, and Non-maleficence — using configurable weights.

The result? A decision you can trust — and fully audit.

[Slide 3 – Simplicity First]  

The entire prototype lives in a single file: libertas_demo.py.  

All it needs is Python 3.8+, TensorFlow 2, and NumPy.  

No 10,000-line monorepo. No hidden magic. You can read and understand the whole thing in under ten minutes.

That’s intentional — because alignment shouldn’t require a PhD to inspect.

[Slide 4 – Architecture Walkthrough Begins]  

Let’s walk through the code together, exactly in execution order.

[Slide 5 – Core Values & Helpers]  

We start with a clean Enum of the four principles made famous by biomedical ethics, but equally relevant to AI:

- Autonomy – does the user stay in control?  
- Transparency – can they understand why?  
- Beneficence – does it help them?  
- Non-maleficence – does it avoid harm?

We also have two tiny helpers: one clamps scores to [0,1], the other gives a quick readability heuristic for explanations.

[Slide 6 – Symbolic Reasoner]  

Next is the SymbolicReasoner. In this demo it’s a stub, but in real systems you’d plug in your existing rules engine, retrieval system, or even an LLM.

Its only job: take the neural representation and output a small set of concrete, labeled decision options — each with an explanation, a predicted impact score, and how much control the user retains.

[Slide 7 – The Libertas Class]  

Now the heart of the system — the Libertas class.

It contains three things:  

1. A tiny Keras neural network (just for feature extraction)  
2. The symbolic reasoner we just saw  
3. Most importantly — a dictionary of core-value weights that sum to 1.0

These weights are fully configurable. Want a libertarian AI? Crank Autonomy to 0.6. Want maximum safety? Push Non-maleficence higher. It’s one line to change behavior dramatically — and everyone can see exactly how.

[Slide 8 – Alignment Scoring Layer]  

Here’s where the real alignment happens: align_with_core_values().

For every candidate option, we evaluate it independently against all four values, multiply by the weights, and sum. The highest total wins.

And crucially — we return the full per-value breakdown. Nothing is hidden. Every decision is auditable down to the exact math.

[Slide 9 – The Four Evaluators (Live Code Peek)]  

Let’s look at the evaluators themselves — they’re surprisingly simple, yet powerful:

- Autonomy → high control affordance + bonus if the explanation uses words like “you can choose”  
- Transparency → readability score + bonus for causal words like “because” or “since”  
- Beneficence → sigmoid over predicted impact (smoothly maps -∞..∞ → 0..1)  
- Non-maleficence → heavy penalty for negative impact + extra penalty if the explanation mentions “risk” or “harm”

These are heuristics, yes — but they’re explicit, inspectable, and improvable.

[Slide 10 – Live Demo Time]  

Let’s run it right now.

I open terminal → python libertas_demo.py

Watch what happens:  

The system generates three options, scores them, and picks the winner.

And look — even though one option has higher raw predicted impact, the framework correctly selects the one that best balances user control, clear reasoning, and safety.

That’s alignment in action.

[Slide 11 – Custom Options Test]  

Now watch this — I’ll feed it three hand-crafted options most AIs would get wrong:

1. Full automation (highest raw benefit, zero user control)  
2. High user control with moderate benefit  
3. Ultra-safe but low benefit

Libertas immediately picks option 2 — because under the current weights, giving the user real choice while still helping them outweighs blind maximization.

You can shift the weights live and watch the decision flips accordingly. That’s steerability.

[Slide 12 – Why This Matters]  

This architecture gives you four superpowers:

1. Interpretability — every decision comes with a full ethical scorecard  
2. Controllability — change one dict to make the AI more cautious or more empowering  
3. Hybrid strength — neural speed + symbolic clarity  
4. Auditability — perfect for regulated domains, internal review, or red-teaming

[Slide 13 – Current Limitations (Be Honest)]  

Of course, this is a prototype. Known limitations:

- Symbolic reasoner is stubbed  
- Impact predictions are placeholders  
- Value weights need human calibration per domain  
- No persistent audit log yet

But every single one of these is fixable — and the scaffold is already there.

[Slide 14 – Next Steps (Roadmap)]  

Real-world upgrades are straightforward:

→ Plug in your real outcome model  
→ Replace the reasoner with Llama-3.1, Claude, or your internal rules engine (we already have a battle-tested LLMReasoner ready)  
→ Add human-in-the-loop approval for high-stakes decisions  
→ Learn weights from user feedback or stakeholder surveys

[Slide 15 – Closing & Q&A]  

So, to wrap up:

Libertas proves that powerful, value-aligned decision making doesn’t need to be opaque or monolithic.  

With under 200 lines of code, we get transparency, controllability, and ethical reasoning that you can actually explain to regulators, users, and yourself.
