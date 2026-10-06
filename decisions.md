# decisions (why????) 

why things in this repo are the way they are. newest at the bottom. 


## 2026-10-03 pgvector over Chroma
Already using Postgres for structured bill data. A second database
means two things to deploy and keep in sync for no real gain.

## 2026-10-03 decimal over float for money
as floats cant represent most decimal amounts exactly, and grading needs exact equality.

## 2026-10-03 dates and money are never strings
"03/15/2026" is ambiguous and you cant do math on it. typed date and Decimal push the parsing problem to the boundary of the system instead of letting it leak everywhere.

## 2026-10-03 field docstrings double as LLM descriptions 
the docstrings in `schema.py` appear in the generated JSON schema, which then gets sent to the model. same source of truth used by both the validator and the prompt.

## 2026-10-03 geberate data first and then render PDF from it
the labels are correct by construction. reading answers back out of a generated PDF would bake the same errors youre trying to measure into the ground truth. 



## Known limitations

- **Synthetic bills are cleaner than real ones.** Every bill from a given
  template is perfectly consistent, which flatters the regex baseline. Real
  providers vary their own layout over time, which is why production parsers
  break constantly.
- **total == sum(line_items) always holds.** On real bills the total often
  includes a previous balance printed outside the charges table. Considering a
  sixth variant where this doesn't hold, so the validation check has to handle
  both.
- **The test set is unseen by the prompts, not by the agent.** Claude Code
  rendered sample PDFs and ran the regex parser across all 200 bills in
  earlier sessions. No prompt was written from test bills, but strict
  isolation wasn't maintained end to end.
- **Temperature model list is hardcoded.** The set of models accepting a
  temperature parameter is a prefix list in client.py and will go stale.