from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.data_pipeline import SEED, generate_synthetic_rows, write_csv


def main() -> None:
    output = ROOT / "data" / "synthetic" / "synthetic_v1_raw.csv"
    write_csv(output, generate_synthetic_rows(seed=SEED))
    print(f"wrote {output} ({SEED=})")


if __name__ == "__main__":
    main()
