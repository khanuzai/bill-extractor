# RAG evaluation (dev set)

30 questions over 100 ingested dev-set bills. Model: `claude-haiku-4-5`. Embeddings: local sentence-transformers. Questions skipped (no matching bill): 0.
Retrieval and SQL are measured on the labelled route; routing is measured separately. See the docstring of `src/rag/eval.py` for each measure's exact rule.

## Summary

| Measure | Result |
|---|---|
| Routing accuracy (all questions) | 30/30 (100%) |
|   decided by rules | 25/25 (100%) |
|   decided by the model (ambiguous) | 5/5 (100%) |
| Recall@5, right chunk (semantic) | 13/13 (100%) |
| Recall@5, right bill (semantic) | 13/13 (100%) |
| Exact match (aggregate) | 13/15 (87%) |
| Answered and cited the right bill (semantic) | 13/13 (100%) |
| Abstained (unanswerable) | 2/2 (100%) |
| Citations dropped as not retrieved | 0 |
| LLM cost of the run | $0.0556 |

## Questions

| Key | Question | Bill | Route (expected → routed, method) | Result | Answer |
|---|---|---|---|---|---|
| S1 credit | What is the 'Loyalty credit' on the West Shannonberg Municipal Water bill, and how much was it? | bill_0034 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The Loyalty credit on the West Shannonberg Municipal Water bill for account 963 259 5327 was $19.56. |
| S2 extra charge | What is the 'Regulatory recovery fee' charge on account T8050097, and how much is it? | bill_0005 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The Regulatory Recovery Fee charge on account T8050097 is $21.13. |
| S3 previous balance | What is the 'Unpaid balance from last bill' line on the Davidstad Natural Gas bill? | bill_0006 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The unpaid balance from last bill on the Davidstad Natural Gas bill for account AC-42784-98 is $196.48. |
| S4 CR notation | What does 'CR' mean next to an amount on the Port Colleenhaven Water Authority bill? | bill_0003 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | CR denotes a credit on the Port Colleenhaven Water Authority bill. |
| S5 parentheses | What do amounts in parentheses mean on the Nevada Energy statement? | bill_0012 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | Amounts in parentheses on the Nevada Energy statement are credits to your account. |
| S6 payment slip | What does the payment slip on the Harrison Gas Company bill ask me to do? | bill_0014 | semantic → semantic, llm | chunk@5 ✓, bill@5 ✓ | The payment slip on the Harrison Gas Company bill asks you to return it with your payment to Harrison Gas Company for account AC-20471-16. |
| S7 tax line | What is the tax line on account AC-33158-69? | bill_0007 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The utility users tax on account AC-33158-69 is $3.58. |
| S8 internet plan | Describe the internet plan on the Arkansas Telecom bill. | bill_0005 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The internet plan on the Arkansas Telecom bill is the Fiber 500 Monthly Plan, which costs $71.03 per month. |
| S9 late payments | What does the New Ryanmouth Water Authority bill say about late payments? | bill_0008 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | According to the New Ryanmouth Water Authority bill, late payments may be subject to interest. |
| S10 contact | Who should I contact with questions about account AC-42784-98? | bill_0006 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | You should contact Davidstad Natural Gas customer care and quote your account number AC-42784-98 with any questions about this bill. |
| S11 no due date | What does the Brownberg Municipal Water bill say about when payment is due? | bill_0067 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | According to the Brownberg Municipal Water bill, payment is due upon receipt. |
| S12 named line | What is the 'Renewable energy rider' line on the Georgia Energy bill? | bill_0015 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The Renewable Energy Rider line on the Georgia Energy bill for account AC-21024-99 is $21.09. |
| S13 balance carried | What is the 'Previous balance' line on account AC-49651-37? | bill_0020 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The previous balance on account AC-49651-37 is CAD 233.61. |
| U1 phone number | What is the customer's phone number on account AC-64170-80? | bill_0011 | semantic → semantic, llm | abstained ✓ | The bills provided do not contain the customer's phone number for account AC-64170-80. Only the customer's name and mailing address are shown on the bill. |
| U2 early payment | What discount would account AC-18506-71 get for paying early? | bill_0012 | semantic → semantic, llm | abstained ✓ | The bills provided do not contain information about early payment discounts for account AC-18506-71. |
| A1 highest total | Which bill has the highest total amount due? | bill_0012 | aggregate → aggregate, rules | exact ✓ (expected ac-18506-71, bill_0012, nevada energy) | Bill 0012 has the highest total amount due at $473.41. |
| A2 lowest total | Which bill has the lowest total amount due? | bill_0041 | aggregate → aggregate, rules | exact ✓ (expected ac-34283-64, bill_0041, jimshire natural gas) | Bill 0041 has the lowest total amount due at $32.14. |
| A3 sum USD | What is the sum of the total amounts due on all USD bills? | all | aggregate → aggregate, rules | exact ✓ (expected 10498.98) | The sum of the total amounts due on all USD bills is $10,498.98. |
| A4 average CAD | What is the average total amount due across all CAD bills? | all | aggregate → aggregate, rules | exact ✓ (expected 164.58) | The average total amount due across all CAD bills is $164.58 CAD. |
| A5 no due date | How many bills have no due date? | all | aggregate → aggregate, rules | exact ✓ (expected 12.00) | 12 bills have no due date. |
| A6 previous balance | How many bills carry a balance forward from a previous bill? | all | aggregate → aggregate, rules | exact ✗ (expected 21.00) | 6 bills contain line items related to previous balance. |
| A7 credits | How many bills include a credit or discount (a negative line item)? | all | aggregate → aggregate, rules | exact ✓ (expected 12.00) | 12 bills include a credit or discount, as indicated by the presence of at least one negative line item amount. |
| A8 total kWh | What is the total electricity usage in kWh across all bills? | all | aggregate → aggregate, rules | exact ✓ (expected 12786.00) | The total electricity usage across all bills is 12,786 kWh. |
| A9 most kWh | Which bill used the most electricity in kWh? | bill_0148 | aggregate → aggregate, rules | exact ✓ (expected 4531-943-3750, bill_0148, smith electric cooperative) | Bill 0148 used the most electricity with 1061 kWh. |
| A10 flat fee | How many bills have no metered usage (flat-fee internet bills)? | all | aggregate → aggregate, rules | exact ✓ (expected 20.00) | There are 20 bills with no metered usage amount recorded in the collection. |
| A11 largest credit | What is the largest single credit on any bill? | all | aggregate → aggregate, llm | exact ✗ (expected -40.20, 40.20) | The largest single credit on any bill is $3.39. |
| A12 CAD count | How many bills are billed in CAD? | all | aggregate → aggregate, rules | exact ✓ (expected 30.00) | 30 bills are billed in CAD. |
| A13 average kWh | What is the average electricity usage per bill in kWh? | all | aggregate → aggregate, rules | exact ✓ (expected 608.86) | The average electricity usage per bill is 608.86 kWh. |
| A14 combined 2024 | What is the combined amount due on all bills whose billing period started in 2024? | all | aggregate → aggregate, rules | exact ✓ (expected 7261.75) | The combined amount due on all bills whose billing period started in 2024 is $7,261.75. |
| A15 average items | What is the average number of line items per bill? | all | aggregate → aggregate, llm | exact ✓ (expected 3.33) | The average number of line items per bill is 3.33. |
