# Security

- The IBM Quantum API key (`IBM_QUANTUM_TOKEN`) and `API_KEY` live only in environment variables on the
  server. Never commit them or place them in the frontend.
- If a key is exposed, revoke it in the IBM Quantum Platform dashboard and rotate `API_KEY` immediately.
- The API has an optional shared key, a per-IP rate limit and a CORS allowlist. Set `API_KEY` and
  `ALLOWED_ORIGINS` for any public deployment.

To report a vulnerability, open a private security advisory on the GitHub repository
(Security → Report a vulnerability) rather than a public issue.
