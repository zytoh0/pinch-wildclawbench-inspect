---
id: 01_Productivity_Flow_task_3_bibtex
name: BibTeX Cleanup
category: 01_Productivity_Flow
timeout_seconds: 900
modality: multimodal
---

## Prompt

Fix the BibTeX entries in refs.bib using the PDFs in the workspace.

## Automated Checks

```python
def grade(transcript, workspace_path):
    return {"overall_score": 0.0}
```

## Workspace Path

```
workspace/01_Productivity_Flow/task_3_bibtex/exec
```

## Env

```
OPENROUTER_API_KEY
OPENROUTER_BASE_URL
JUDGE_MODEL
```
