"""Fake credentials for tests.

Each value is assembled when the module is imported, so no key-shaped string literal appears
in the source. That keeps secret scanners (and the repo scan in test_security.py) from
mistaking test data for a leaked key. None of these are real.
"""

FAKE_OPENAI_KEY = "sk-" + "test" * 7
FAKE_OPENROUTER_KEY = "sk-or-v1-" + "0123456789abcdef" * 2
FAKE_GROQ_KEY = "gsk_" + "Fake123456" * 3
FAKE_AWS_KEY_ID = "AKIA" + "TESTTESTTEST0000"
FAKE_BEARER = "Bearer " + "abcdefghijklmnop12345"
FAKE_GITHUB_TOKEN = "ghp_" + "a1b2c3" * 6
FAKE_JWT = ".".join([
    "eyJ" + "hbGciOiJIUzI1NiJ9",
    "eyJ" + "zdWIiOiIxMjM0NTY3OA",
    "sig" + "nature123",
])
FAKE_PRIVATE_KEY = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKfake\n-----END " \
    + "RSA PRIVATE KEY-----"
