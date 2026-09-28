---
doc_id: SPEC-003
title: Mobile Bill Payments
department: retail-banking
document_type: product_spec
access_level: internal
created_date: 2026-01-19
tags: [bill-payments, mobile-app, product-spec, billers, scheduled-payments]
---

## Product Summary

Mobile Bill Payments lets Commercial Bank customers pay utility, telecom, insurance, school and credit card bills from the mobile banking app. Mobile Bill Payments replaces the older internet banking bill pay screens and adds saved billers, bill reminders and scheduled payments. The product owner is the retail digital team.

## Supported Billers

At launch, Mobile Bill Payments supports about 120 billers onboarded through the PayCore biller hub, grouped into categories: electricity and water, telecom and internet, insurance, education, and other banks' credit cards. New billers are added by the payments operations team without an app release. Each biller defines its own reference number format, which Mobile Bill Payments validates before payment.

## Customer Experience

In Mobile Bill Payments, the customer picks a biller, enters or scans the bill reference, and the app fetches the outstanding amount where the biller supports bill presentment. The customer can pay the full or a partial amount from a current or savings account, or by card using an in-app card token. Customers can save a biller with a nickname, turn on due-date reminders three days before the due date, and schedule recurring payments monthly.

## Limits and Fees

Mobile Bill Payments is free for customers for all billers. The daily Mobile Bill Payments limit is 500,000 per customer, separate from the InstaPay limit. A single bill payment cannot exceed 250,000. Scheduled bill payments that would exceed the daily limit are held and the customer is notified.

## Processing Flow

A Mobile Bill Payments request goes from the mobile app to the mobile-bff, then to payments-api, which reserves funds in payments-ledger and sends the payment to the PayCore biller hub. The biller confirmation is recorded in payments-ledger and published to the payment-events Kafka topic, which triggers the payment receipt notification. If the biller hub does not confirm within 60 seconds, the payment is marked pending and resolved in the next reconciliation cycle.

## Failure Handling

Mobile Bill Payments must never debit a customer twice. Each bill payment carries an idempotency key. If a payment is debited but the biller rejects it, the funds are returned automatically within one business day and the customer receives a push notification. Scheduled payments that fail are retried once after 4 hours.

## Non-Functional Requirements

Mobile Bill Payments targets 99.9 percent availability, a p95 payment time under 8 seconds, and must handle the month-end and salary-day peak when bill payment volume triples.

## Success Metrics

Mobile Bill Payments success is measured by moving 70 percent of bill payments from branch and internet banking to mobile within a year, and by a payment success rate above 99 percent.
