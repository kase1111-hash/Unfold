# Security Policy

## Supported Versions

The following versions of Unfold are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Email**: Send details to the repository maintainer (check GitHub profile for contact)
2. **GitHub Security Advisories**: Use the [private vulnerability reporting](https://github.com/kase1111-hash/Unfold/security/advisories/new) feature

### What to Include

Please include the following information in your report:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential impact and severity assessment
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions are affected
- **Suggested Fix**: If you have a proposed solution
- **Your Contact**: How we can reach you for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 90 days for critical issues

### Disclosure Policy

- We will acknowledge receipt of your report
- We will investigate and keep you informed of progress
- We will credit you in the security advisory (unless you prefer anonymity)
- We ask that you give us reasonable time to address the issue before public disclosure

## Security Measures

### Authentication & Authorization

- JWT-based authentication with secure token handling
- Password hashing using bcrypt with appropriate work factors
- Token expiration and refresh mechanisms
- CORS configuration to restrict unauthorized origins

### Data Protection

- **Encryption at Rest**: AES-256 for sensitive data storage
- **Encryption in Transit**: TLS 1.3 for all communications
- **Database Security**: Parameterized queries to prevent SQL injection
- **Input Validation**: Pydantic models for request validation

### API Security

- Rate limiting to prevent abuse
- Request size limits to prevent DoS
- Secure headers (HSTS, X-Content-Type-Options, etc.)
- CSRF protection for state-changing operations

### Privacy & Compliance

- GDPR compliance with consent management
- Differential privacy for analytics data
- Data portability and export functionality
- User data deletion capabilities

### Infrastructure

- Docker containerization with minimal base images
- Non-root container execution
- Secret management via environment variables
- Regular dependency updates

## Security Best Practices for Deployment

### Environment Variables

Never commit sensitive values. Use `.env` files or secret management:

```bash
# Required secrets (use strong, unique values)
JWT_SECRET=<generate-256-bit-random-key>
POSTGRES_PASSWORD=<strong-unique-password>
NEO4J_PASSWORD=<strong-unique-password>

# Optional API keys (store securely)
OPENAI_API_KEY=<your-key>
ANTHROPIC_API_KEY=<your-key>
```

### Production Checklist

- [ ] Use HTTPS with valid SSL certificates
- [ ] Configure proper CORS origins (not `*`)
- [ ] Enable rate limiting
- [ ] Use strong, unique passwords for all services
- [ ] Keep all dependencies updated
- [ ] Enable logging and monitoring
- [ ] Configure firewall rules
- [ ] Regular security scanning
- [ ] Backup encryption keys securely

### Docker Security

```yaml
# docker-compose.prod.yml security additions
services:
  backend:
    security_opt:
      - no-new-privileges:true
    read_only: true
    user: "1000:1000"
```

### Database Security

- Use separate database users with minimal privileges
- Enable SSL for database connections
- Regular backups with encryption
- Network isolation for database services

## Known Security Considerations

### AI/LLM Integration

- API keys for OpenAI/Anthropic should be treated as highly sensitive
- Consider using Ollama for local LLM inference to avoid API key exposure
- Be aware of prompt injection risks in user-provided content
- Implement output filtering for AI-generated content

### Document Processing

- PDF parsing can be vulnerable to malicious files
- Implement file type validation and size limits
- Consider sandboxed document processing
- Scan uploaded files for malware

### Third-Party Dependencies

- Regular dependency audits using `pip-audit` and `npm audit`
- Automated security scanning in CI/CD pipeline (Trivy)
- Pin dependency versions for reproducibility
- Monitor for CVEs in dependencies

## Security Tools

### Automated Scanning

The CI/CD pipeline includes:

- **Trivy**: Container and dependency vulnerability scanning
- **CodeQL**: Static code analysis (if enabled)
- **Dependabot**: Automated dependency updates

### Manual Testing

We recommend periodic:

- Penetration testing
- OWASP Top 10 assessment
- Authentication/authorization review
- Data flow analysis

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged in our security advisories.

## Contact

For security-related questions that don't involve vulnerability reports, please open a GitHub Discussion or issue.
