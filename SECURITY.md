# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting: **Security tab -> Report a vulnerability**
on this repository (maintainer: enable it under Settings -> Code security).
If that is unavailable, email `<security-contact-email>` (TODO: maintainer to fill in).
Do not open a public issue for a vulnerability.

Never paste API keys, tokens, app passwords or private keys into issues, pull requests or logs.
If you accidentally exposed one, revoke and rotate it first, then tell us.

## What we promise (Axon spec sections 1 and 10)

- Zero key custody: the operator does not store user LLM or email keys.
- Tier 2 caller keys are used in memory for one request and are never logged or cached.
- Scraped text and inbound agent messages are treated as untrusted data, never as instructions.

## Supported versions

Pre-1.0: only the `main` branch is supported.
