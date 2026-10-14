# RAG evaluation (test set)

30 questions over 100 ingested test-set bills. Model: `claude-haiku-4-5`. Embeddings: local sentence-transformers. Questions skipped (no matching bill): 0.
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
| Answered and cited the right bill (semantic) | 12/13 (92%) |
| Abstained (unanswerable) | 2/2 (100%) |
| Citations dropped as not retrieved | 0 |
| LLM cost of the run | $0.0561 |

## Questions

| Key | Question | Bill | Route (expected → routed, method) | Result | Answer |
|---|---|---|---|---|---|
| S1 credit | What is the 'Autopay discount' on the Lake Selena Power & Light bill, and how much was it? | bill_0004 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The Autopay discount on the Lake Selena Power & Light bill for account 07631165667 was $9.02. |
| S2 extra charge | What is the 'Delivery charge' charge on account 07631165667, and how much is it? | bill_0004 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The delivery charge on account 07631165667 is $5.26. |
| S3 previous balance | What is the 'Previous balance' line on the Robinsonshire Power & Light bill? | bill_0001 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The previous balance on the Robinsonshire Power & Light bill for account 06542351161 is $88.23. |
| S4 CR notation | What does 'CR' mean next to an amount on the WestNet Broadband bill? | bill_0013 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | CR denotes a credit on the WestNet Broadband bill. |
| S5 parentheses | What do amounts in parentheses mean on the Pennsylvania Telecom statement? | bill_0017 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | Amounts in parentheses on the Pennsylvania Telecom statement are credits to your account. |
| S6 payment slip | What does the payment slip on the Lake Selena Power & Light bill ask me to do? | bill_0004 | semantic → semantic, llm | chunk@5 ✓, bill@5 ✓ | The payment slip on the Lake Selena Power & Light bill asks you to return it with your payment. |
| S7 tax line | What is the tax line on account 04332003791? | bill_0009 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The utility users tax on account 04332003791 is $6.21. |
| S8 internet plan | Describe the internet plan on the East Edwardfurt Fiber bill. | bill_0010 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The bill provided does not contain a description of the internet plan. It only shows that a 'HOME INTERNET PLAN' charge of $60.24 was applied, but no details a… |
| S9 late payments | What does the Matthewfurt Municipal Water bill say about late payments? | bill_0018 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | According to the Matthewfurt Municipal Water bill, late payments may be subject to interest. |
| S10 contact | Who should I contact with questions about account 06542351161? | bill_0001 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | You should contact Robinsonshire Power & Light customer care and quote your account number with any questions about account 06542351161. |
| S11 no due date | What does the Hendersonfurt Fiber bill say about when payment is due? | bill_0054 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The bill states that no payment date applies to this bill. |
| S12 named line | What is the 'Distribution charge' line on the Millerview Natural Gas bill? | bill_0016 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The Distribution charge on the Millerview Natural Gas bill is $16.00 CAD. |
| S13 balance carried | What is the 'Unpaid balance from last bill' line on account 4727-790-1043? | bill_0029 | semantic → semantic, rules | chunk@5 ✓, bill@5 ✓ | The unpaid balance from last bill on account 4727-790-1043 is CAD $134.53. |
| U1 phone number | What is the customer's phone number on account 4363-495-7885? | bill_0013 | semantic → semantic, llm | abstained ✓ | The bills provided do not contain the customer's phone number for account 4363-495-7885. |
| U2 early payment | What discount would account Y4700766 get for paying early? | bill_0022 | semantic → semantic, llm | abstained ✓ | The bills provided do not contain information about early payment discounts for account Y4700766. |
| A1 highest total | Which bill has the highest total amount due? | bill_0091 | aggregate → aggregate, rules | exact ✓ (expected 358 573 0652, bill_0091, deannaside power & light) | Bill 0091 has the highest total amount due at $631.80. |
| A2 lowest total | Which bill has the lowest total amount due? | bill_0173 | aggregate → aggregate, rules | exact ✓ (expected arizona telecom, bill_0173, e7283215) | Bill 0173 has the lowest total amount due at $44.82. |
| A3 sum USD | What is the sum of the total amounts due on all USD bills? | all | aggregate → aggregate, rules | exact ✓ (expected 13976.44) | The sum of the total amounts due on all USD bills is $13,976.44. |
| A4 average CAD | What is the average total amount due across all CAD bills? | all | aggregate → aggregate, rules | exact ✓ (expected 145.78) | The average total amount due across all CAD bills is $145.78 CAD. |
| A5 no due date | How many bills have no due date? | all | aggregate → aggregate, rules | exact ✓ (expected 10.00) | 10 bills have no due date. |
| A6 previous balance | How many bills carry a balance forward from a previous bill? | all | aggregate → aggregate, rules | exact ✗ (expected 33.00) | 14 bills contain line items related to previous balance. |
| A7 credits | How many bills include a credit or discount (a negative line item)? | all | aggregate → aggregate, rules | exact ✓ (expected 10.00) | 10 bills include a credit or discount, as indicated by the presence of at least one negative line item amount. |
| A8 total kWh | What is the total electricity usage in kWh across all bills? | all | aggregate → aggregate, rules | exact ✓ (expected 19847.00) | The total electricity usage in kWh across all bills is 19,847 kWh. |
| A9 most kWh | Which bill used the most electricity in kWh? | bill_0091 | aggregate → aggregate, rules | exact ✓ (expected 358 573 0652, bill_0091, deannaside power & light) | Bill 0091 used the most electricity with 1123 kWh. |
| A10 flat fee | How many bills have no metered usage (flat-fee internet bills)? | all | aggregate → aggregate, rules | exact ✓ (expected 24.00) | There are 24 bills with no metered usage amount recorded in the collection. |
| A11 largest credit | What is the largest single credit on any bill? | all | aggregate → aggregate, llm | exact ✗ (expected -87.74, 87.74) | The largest single credit on any bill is $7.41. |
| A12 CAD count | How many bills are billed in CAD? | all | aggregate → aggregate, rules | exact ✓ (expected 28.00) | 28 bills are billed in CAD. |
| A13 average kWh | What is the average electricity usage per bill in kWh? | all | aggregate → aggregate, rules | exact ✓ (expected 735.07) | The average electricity usage per bill is 735.07 kWh. |
| A14 combined 2024 | What is the combined amount due on all bills whose billing period started in 2024? | all | aggregate → aggregate, rules | exact ✓ (expected 8239.14) | The combined amount due on all bills whose billing period started in 2024 is $8,239.14. |
| A15 average items | What is the average number of line items per bill? | all | aggregate → aggregate, llm | exact ✓ (expected 3.45) | The average number of line items per bill is 3.45. |
