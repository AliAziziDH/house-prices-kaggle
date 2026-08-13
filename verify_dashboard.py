import sys
from src.ui.app import load_data

def main():
    df, app_mode = load_data()
    print(f"Data length: {len(df)}")
    print(f"App Mode: {app_mode}")
    assert len(df) == 1459, f"Expected 1459 rows, got {len(df)}"
    assert app_mode == "Production Mode", f"Expected 'Production Mode', got '{app_mode}'"
    print("✅ Dashboard transition verified successfully!")

if __name__ == '__main__':
    main()
