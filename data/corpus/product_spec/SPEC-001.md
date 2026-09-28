---
doc_id: SPEC-001
title: InstaPay Instant Payments
department: payments
document_type: product_spec
access_level: internal
created_date: 2026-09-07
tags: [instapay, instant-payments, product-spec, limits, nationalswitch]
---

## Product Summary

InstaPay is Commercial Bank's instant payments product. InstaPay lets retail and small business customers send money to any account at Commercial Bank or at another participating bank through NationalSwitch, with the funds credited to the recipient within 10 seconds, 24 hours a day, 7 days a week. The product owner for InstaPay is Nadeesha Silva, product manager for payments. All amounts in this specification are in the bank's base currency.

## Customer Experience

InstaPay is available in the mobile banking app and internet banking. The customer selects a saved payee or enters an account number or registered mobile number, and InstaPay performs a name enquiry so the customer can confirm the recipient name before paying. The customer sees a confirmation screen with the fee, then a success receipt within 10 seconds. If InstaPay cannot complete within 30 seconds, the payment is shown as pending and the customer receives a push notification with the final result.

## InstaPay Limits

InstaPay limits: the maximum InstaPay transaction amount is 1,000,000 per payment, and the default daily InstaPay limit is 2,000,000 per customer. Customers can lower their InstaPay daily limit in the app, and can raise it up to 5,000,000 after step-up verification with an OTP. Small business customers have a daily InstaPay limit of 10,000,000. The first payment to a new payee is capped at 100,000 for the first 24 hours. InstaPay limits are held in payments-api configuration in minor currency units, and any limit change must pass range validation and a canary.

## InstaPay Fees

InstaPay payments between Commercial Bank accounts are free. InstaPay payments to other banks cost 25.00 per payment for retail customers and 15.00 for small business customers on the business package. Fees are shown before confirmation and are never charged on failed payments.

## Processing Flow

An InstaPay payment flows from the mobile-bff to payments-api, which checks limits and fees, requests a fraud-engine score, reserves funds in payments-ledger, and sends the payment to NationalSwitch through the nationalswitch-connector for other banks. On confirmation, payments-ledger posts the payment and publishes to the payment-events Kafka topic for settlement and notifications.

## Bulk Payee Validation

Small business customers can upload up to 200 payees at once, and InstaPay validates all payees with a single batched ledger query and batched name enquiries. Bulk payee validation must stay within its connection budget on payments-ledger and must be included in the salary-day load test.

## Non-Functional Requirements

InstaPay targets 99.95 percent monthly availability, a p95 end-to-end time under 5 seconds, and throughput of 400 payments per second at peak. InstaPay must degrade gracefully: when NationalSwitch is unavailable, InstaPay payments to other banks fail fast with a clear message while payments inside Commercial Bank continue.

## Out of Scope

InstaPay does not support international transfers, scheduled future-dated payments or payments from credit card accounts in this release.
