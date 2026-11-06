# Example datasets

Templates for bringing your own data to Studio's recipes. Choose "JSONL file" as the recipe's data and
give the file's path. These are tiny and synthetic: they show the format, and train only well enough
to check the plumbing. Real use wants hundreds to thousands of lines.

| File | Recipe | One line |
|---|---|---|
| `critic-example.jsonl` | Jev-style critic | `{"state", "question", "type": "choice" \| "score" \| "noul", "options": {name: description}, "answer"}` |
| `probe-example.jsonl` | Salience probes | `{"text", "labels": [...]}` |
| `pairs-example.jsonl` | Fine-tune EmbeddingGemma 2 | `{"query", "document"}` |

How the critic reads its file:
- **Score:** a Score question's options are its levels, lowest first.
- **Yes/no:** a `noul` question needs no options. Its answer is `true` or `false`.
- **Questions:** the same question text on many lines is one question.
- **Splits:** states (and probe texts) are split 80/10/10 into train, validation and test by a hash of
  the text, so the same state never lands in two splits.
