# Meridian Outfitters Incident Response Plan

This plan covers security incidents affecting Meridian systems or data. It works with the Information Security Policy, which defines the controls, and the Data Retention and Privacy document, which defines notification duties.

## What Counts as an Incident

An incident is any event that compromises, or credibly threatens to compromise, the confidentiality, integrity, or availability of Meridian systems or data.

Examples that qualify: a phishing email that someone entered credentials into, malware on an endpoint, an exposed access key, unauthorized access to a customer account, a lost unencrypted device, a ransomware demand, a website outage caused by attack rather than error.

Examples that do not qualify on their own: a blocked phishing email nobody interacted with, a failed login from an unrecognized location that MFA stopped, a routine outage from a deployment error. These are logged but not run through this plan.

When in doubt, report it. A false positive costs an hour. An unreported incident costs considerably more.

## Severity Levels

**SEV1** — Confirmed unauthorized access to customer personal data, payment systems, or production infrastructure. Ransomware affecting any production system. Response begins immediately at any hour.

**SEV2** — Credible threat to a production system without confirmed data access. Compromise of a single employee account with elevated access. Malware that endpoint protection did not contain. Response begins within 1 hour during business hours, within 4 hours otherwise.

**SEV3** — Compromise of a single standard employee account. Exposed credential with no evidence of use. Non-production system compromise. Response begins the next business day.

Severity is assigned by the on-call security engineer and can be raised by anyone. It is lowered only by the incident commander, and the reason is recorded.

## Roles

The **incident commander** runs the response and is the single decision maker. The role is not necessarily the most senior person present, and the commander explicitly names a successor when handing off.

The **communications lead** handles all internal and external messaging. No one else speaks publicly about an active incident, which prevents contradictory statements reaching customers.

The **scribe** maintains the timeline: what was observed, what was decided, what was done, and when. The timeline is written during the incident rather than reconstructed afterward, because reconstruction produces a tidier and less accurate account.

**Subject matter experts** are pulled in as needed and released when their part is done.

## Response Phases

**Detect and report.** Anyone reports to security@meridianoutfitters.example or the security channel. For SEV1 the reporter also calls the on-call number, because email is not a paging mechanism.

**Contain.** Stop the spread before understanding it fully. Isolate affected hosts, revoke sessions and keys, disable affected accounts. Containment decisions that disrupt the business are the incident commander's to make and do not wait for executive approval.

**Eradicate.** Remove the mechanism. Rotate every credential that was exposed or could have been. Rebuild compromised hosts from a known-good image rather than cleaning them in place.

**Recover.** Restore service and monitor closely. Recovery is not complete when service resumes; it is complete when a defined monitoring window passes without recurrence, normally 72 hours for SEV1.

**Review.** A written postmortem within 10 business days for SEV1 and SEV2.

## Preserving Evidence

Do not power off a compromised host. Capture memory and disk images first, because powering off destroys volatile evidence that often carries the only record of what ran.

Do not delete logs, mailboxes, or files that may be relevant, even where retention would otherwise allow it. An incident places an implicit hold on related data until the incident commander releases it.

Do not communicate about a suspected internal compromise on the systems that may be compromised. The response has an out-of-band channel for this reason.

## Postmortems

Postmortems are blameless. The purpose is to find the conditions that allowed the incident, not the person who was closest to it. A postmortem naming an individual as a root cause is sent back for rework.

Every postmortem states the timeline, the customer impact, the contributing factors, and the actions with named owners and dates. Actions without an owner and a date are not actions.

Postmortems are readable by all employees. Incidents involving personal data are redacted for the wide audience and kept in full in the security record.

## Testing

The plan is exercised twice a year with a tabletop, and once a year with a technical exercise in which a credential is deliberately exposed in a staging environment to test detection.

An exercise that the team passes easily is treated as a signal that the scenario was too gentle, and the next one is made harder.
