# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly by emailing brij.pydata@gmail.com. Do not open a public issue.

## Secret Management

**NEVER** hardcode API keys, tokens, or credentials in source code or configuration files.

### How to manage secrets in this project:

1. Copy `.env.example` to `.env`
2. Add your actual API keys to `.env`
3. The `.env` file is git-ignored and will never be committed

```bash
cp .env.example .env
# Edit .env with your actual keys
```

### What NOT to do:

```python
# BAD - hardcoded key
client = OpenAI(api_key="sk-abc123...")

# GOOD - from environment
from config import get_api_key
api_key = get_api_key("openai")
```

## Security Measures in This Project

1. **Environment-based secrets**: All API keys are loaded from environment variables via `config.get_api_key()`
2. **Log sanitization**: The logging system (`src/utils/logger.py`) automatically redacts API keys and tokens from log output
3. **Input validation**: All user inputs are validated before being sent to LLM APIs
4. **Error message sanitization**: Error messages are scrubbed of sensitive data before being raised or logged
5. **Cache file permissions**: Cached API responses are stored with restricted file permissions (0600)
6. **`.gitignore`**: Prevents accidental commit of `.env` files, cache data, and other sensitive files
7. **Rate limiting**: Protects against accidental runaway API calls that could incur large costs

## Dependencies

Keep dependencies updated to patch known vulnerabilities:

```bash
pip install --upgrade -r requirements.txt
```

Consider using `pip-audit` to check for known vulnerabilities:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```
