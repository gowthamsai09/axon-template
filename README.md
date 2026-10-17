# axon-template

GitHub **template repository** for the per-user pipeline of [Axon](https://github.com/gowthamsai09/Axon)
(Plane 2). Click **Use this template** (not Fork) and make your copy **private**. Spec: sections 4.2, 6, 9.

**Phase 0 status:** `profile.yaml` model and validation, task-result model, prompts, and workflow
stubs. The workflows do nothing yet and are disabled in the template repo itself.

```bash
python -m digest.cli validate-profile profile.yaml
```
Secrets (LLM key, SMTP, bot tokens) go in **Settings -> Secrets and variables -> Actions**, never in files.
