# Meridian Outfitters Information Security Policy

This policy applies to all employees, contractors, and temporary staff who access Meridian systems or data. Violations may result in disciplinary action up to termination.

## Passwords and Authentication

All Meridian accounts require passwords of at least 14 characters. Passwords must not be reused across systems or shared with anyone, including IT staff. The company password manager (1Password business vault) must be used to generate and store credentials.

Multi-factor authentication (MFA) is mandatory on all company systems that support it, including email, Workday, Slack, and the VPN. Authenticator apps or hardware keys are required; SMS codes are permitted only as a backup method.

Accounts are locked after 5 failed login attempts and require an IT service desk unlock.

## Device Security

Company laptops must run the managed endpoint agent, have full-disk encryption enabled, and lock automatically after 5 minutes of inactivity. Operating system updates must be installed within 7 days of release; the endpoint agent enforces this.

Personal devices may access company email and Slack only if enrolled in mobile device management (MDM). Enrolled devices must have a passcode and remote-wipe enabled.

Lost or stolen devices must be reported to the IT service desk within 4 hours of discovery so the device can be remotely wiped.

## Data Classification and Handling

Meridian data is classified in three tiers:

- Public: marketing materials, published prices. No restrictions.
- Internal: policies, org charts, sales dashboards. Do not share outside the company.
- Restricted: customer personal data, payment data, employee records, credentials. Access on a need-to-know basis only, never stored on personal devices, and never sent by email; use the secure file share.

Customer payment card data is handled exclusively inside our PCI-compliant payment processor. Employees must never write down, photograph, or ask a customer to email card numbers.

## Customer Data Privacy

Customer personal data may be accessed only to perform your job duties. Looking up a customer's records without a business reason is prohibited and logged.

Data subject requests (access or deletion requests from customers) must be forwarded to privacy@meridianoutfitters.example within 2 business days. Only the privacy team responds to these requests.

Customer data is retained for 5 years after the last transaction, after which it is anonymized. Marketing email lists are purged of addresses that have been unsubscribed for more than 30 days.

## Acceptable Use

Company systems are for business use; incidental personal use is permitted if it does not interfere with work or violate policy. Employees must not install unapproved software on company laptops, use company systems to access illegal content, or connect company devices to untrusted public Wi-Fi without the VPN.

Use of AI tools with company data is permitted only through the approved enterprise accounts listed in the IT tool catalog. Pasting Restricted data into unapproved AI tools is a reportable incident.

## Phishing and Social Engineering

Report suspicious emails using the Report Phish button in Outlook; do not click links or open attachments. The security team runs simulated phishing exercises quarterly; employees who fail 3 simulations in a rolling year must complete additional training.

No one at Meridian, including IT and executives, will ever ask you for your password. Treat any such request as an attack and report it.

## Incident Reporting

Suspected security incidents (malware, data exposure, lost devices, compromised accounts) must be reported to security@meridianoutfitters.example or the 24/7 hotline at extension 4911 within 24 hours of discovery. Faster is better; there is no penalty for good-faith reporting, even if the incident resulted from your own mistake.

The security team triages all reports within 4 business hours and coordinates response, containment, and any required customer or regulator notifications.

## Vendor and Third-Party Access

Vendors may access Meridian systems only through sponsored accounts with an assigned internal owner, scoped to the minimum access needed, and reviewed quarterly. Vendor accounts are disabled automatically after 30 days of inactivity.

## Security Training

All staff complete security awareness training within 14 days of hire and annually thereafter. Engineers additionally complete secure coding training each year. Completion is tracked in Workday and is a condition of continued system access.
