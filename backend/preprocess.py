import re

BANK_NOISE_PATTERN = re.compile(
    r'\b(upi|neft|imps|rtgs|pos|ach|dr|cr|txn|ref|inr|rs|amount|hdfc|icici|sbi|axis|kotak|paytm|gpay|phonepe|ltd|pvt)\b'
)

def clean_transaction_text(text: str) -> str:
    text = str(text).lower()
    # Remove hex reference codes like Ref:0f8f77bb
    text = re.sub(r'ref:[a-z0-9]+', ' ', text)
    # Remove banking prefixes and currency tags
    text = BANK_NOISE_PATTERN.sub(' ', text)
    # Remove numbers, pipes, slashes, and punctuation
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()