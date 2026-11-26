from backend.preprocess import clean_transaction_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "financial_transaction_train.csv"))
test_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "financial_transaction_test.csv"))