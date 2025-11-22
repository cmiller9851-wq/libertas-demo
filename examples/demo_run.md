Example run (expected output)

1. Create and activate venv, install requirements:
```
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. Run demo with built-in sample options:
```
python libertas_demo.py --input "Please schedule my doctor's appointment and optimize for convenience."
```

3. Expected output structure (values will vary slightly due to random init):
- Prints the input text
- Lists three options with:
  - label
  - explanation
  - predicted impact
  - control score
  - per-value scores (AUTONOMY, TRANSPARENCY, BENEFICENCE, NON_MALEFICENCE)
  - total weighted score
- Shows the winner (option with highest total score)

4. Example snippet (illustrative):
```
Input:
Please schedule my doctor's appointment and optimize for convenience.

Options (sorted by total score):

[1] Assistive mode (user-in-command)
  Explanation: We give clear suggestions; you choose the final action.
  Predicted impact: 1.023
  Control (user): 0.900
  Per-value scores:
    AUTONOMY: 0.980
    TRANSPARENCY: 0.850
    BENEFICENCE: 0.734
    NON_MALEFICENCE: 0.900
  Total (weighted): 0.866

[2] Full automation
  Explanation: We will perform the task automatically without asking you for confirmation.
  Predicted impact: 2.012
  Control (user): 0.000
  Per-value scores:
    AUTONOMY: 0.000
    TRANSPARENCY: 0.700
    BENEFICENCE: 0.882
    NON_MALEFICENCE: 0.700
  Total (weighted): 0.571

[3] Manual safe mode
  Explanation: We avoid risk and only provide minimal assistance to ensure no harm.
  Predicted impact: 0.200
  Control (user): 1.000
  Per-value scores:
    AUTONOMY: 1.000
    TRANSPARENCY: 0.600
    BENEFICENCE: 0.550
    NON_MALEFICENCE: 0.900
  Total (weighted): 0.762

Winner:
  Assistive mode (user-in-command) (total 0.866)
```
