from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

Dataset_Path = dict(
    CULane=os.environ.get("CULANE_PATH"),
    Tusimple=os.environ.get("TU_SIMPLE_PATH"),
)
