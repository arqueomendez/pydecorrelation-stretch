# Security Policy

## Supported Versions

Only the latest major version of `pydecorrelation-stretch` typically receives security updates. We strongly encourage users to always use the latest available version.

| Version | Supported          | Notes                                  |
| :-----: | :----------------: | :------------------------------------- |
|  0.4.x  | :white_check_mark: | Current stable release (Active)       |
|  0.2.x  |        :x:         | End of Life (EOL)                     |
| < 0.2.0 |        :x:         | End of Life (EOL) - Upgrade required   |

## Reporting a Vulnerability

We value the security of our community and collaborators. If you believe you have found a security vulnerability in `pydecorrelation-stretch`, please report it to us as outlined below.

**Do NOT report security vulnerabilities through public GitHub issues.**

### How to Report properly

1.  **Private Report**: Please use the **GitHub Security Advisories** feature allowing for private vulnerability reporting (if enabled on this repository) or email the maintainer directly at `victor.mendez@uc.cl`.
2.  **Details Required**:
    *   A description of the vulnerability.
    *   Steps to reproduce the issue (proof of concept code or scripts are highly appreciated).
    *   The version(s) of `pydecorrelation-stretch` affected.
    *   The potential impact of the vulnerability.

### Response Timeline

*   **Acknowledgment**: We aim to confirm receipt of your report within **48 hours**.
*   **Assessment**: We will assess the severity and impact within **1 week**.
*   **Fix**: We aim to release a patch or workaround as soon as possible, prioritizing critical issues.

## Scope

### In Scope
*   Code execution vulnerabilities via malicious image inputs (e.g., buffer overflows in dependent libraries triggered by our code).
*   Improper handling of temporary files or permissions.
*   Logic flaws permitting unintended data modification.

### Out of Scope
*   Vulnerabilities in third-party libraries (e.g., `opencv-python`, `numpy`) should be reported to their respective maintainers, though we welcome heads-up notices to update our dependencies.
*   Denial of Service (DoS) attacks requiring large amounts of resources unless it is caused by a specific implementation flaw (e.g., algorithmic complexity attacks).
*   Issues related to running the software on compromised or insecure systems/hardware.

## Security Best Practices for Users

*   **Input Validation**: Always source images from trusted locations when running in automated batch modes.
*   **Updates**: Keep `pydecorrelation-stretch` and its dependencies updated (`uv sync` or `pip install --upgrade ...`).
*   **Environment**: Run the library in isolated environments (e.g., Virtual Environments, Containers) when processing untrusted data.

## Disclosure Policy

We follow a **Coordinated Disclosure** policy:
*   We ask that you give us reasonable time to investigate and mitigate the issue before releasing public information.
*   We will mention your contribution in the security advisory or release notes (with your permission).

---
*Policy last updated: 2026-01-06*
