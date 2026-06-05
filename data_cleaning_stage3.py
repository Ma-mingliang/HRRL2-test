import os
from datetime import datetime

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================== 可按需调整的配置 ==================
ROOT_DIR = os.path.join(PROJECT_ROOT, "model")
PREFIX = "stage3_complex_mlp"
INPUT_FILENAME = "rl_training_errors.csv"

AGG_DIR = os.path.join(ROOT_DIR, "stage3_data")
REPORT_CSV = os.path.join(ROOT_DIR, "batch_cleaning_report_stage3.csv")
# =====================================================

REQUIRED_COLS = [
    "episode",
    "steps",
    "completed",
    "completion_steps",
    "termination_reason",
    "progress_percent",
    "avg_lateral",
    "std_lateral",
    "max_lateral",
    "avg_course",
    "std_course",
    "max_course",
    "avg_tilt",
    "std_tilt",
    "max_tilt",
    "avg_velocity",
    "avg_action_change",
    "reward",
]

TIMEOUT_REASONS = {"time_out", "timeout"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按第三阶段规则清洗，并返回清洗后的 df 与统计信息。"""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLS].copy()

    for col in ["episode", "steps", "completed", "completion_steps", "reward"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_rows = len(df)

    mask_drop1 = df["steps"] <= 100
    drop1_count = int(mask_drop1.sum())
    df = df[~mask_drop1].copy()

    mask_drop2 = (
        (df["steps"] > 2300)
        & (df["termination_reason"].isin(TIMEOUT_REASONS))
        & (df["reward"] < 0)
    )
    drop2_count = int(mask_drop2.sum())
    df = df[~mask_drop2].copy()

    mask_complete = df["steps"] >= 1800
    modify_count = int(mask_complete.sum())
    df.loc[mask_complete, "completed"] = 1
    df.loc[mask_complete, "termination_reason"] = "task_complete"
    df.loc[mask_complete, "completion_steps"] = df.loc[mask_complete, "steps"]

    df["completion_index"] = 0
    completed_rows = df["completed"] == 1
    df.loc[completed_rows, "completion_index"] = range(1, int(completed_rows.sum()) + 1)

    df["new_episode"] = range(1, len(df) + 1)

    ordered_cols = [
        "episode",
        "new_episode",
        "steps",
        "completed",
        "completion_index",
        "completion_steps",
        "termination_reason",
        "progress_percent",
        "avg_lateral",
        "std_lateral",
        "max_lateral",
        "avg_course",
        "std_course",
        "max_course",
        "avg_tilt",
        "std_tilt",
        "max_tilt",
        "avg_velocity",
        "avg_action_change",
        "reward",
    ]
    df = df[ordered_cols]

    after_rows = len(df)
    completed_count = int((df["completed"] == 1).sum())

    stats = {
        "rows_before": before_rows,
        "rows_after": after_rows,
        "dropped_steps_le_100": drop1_count,
        "dropped_timeout_steps_gt_2300_reward_lt_0": drop2_count,
        "modified_steps_ge_1800_to_complete": modify_count,
        "completed_count_after": completed_count,
    }
    return df, stats


def main():
    ensure_dir(AGG_DIR)

    report_rows = []
    processed = 0
    skipped = 0
    errors = 0

    for dirpath, _, _ in os.walk(ROOT_DIR):
        folder_name = os.path.basename(dirpath)

        if not folder_name.startswith(PREFIX):
            continue

        input_path = os.path.join(dirpath, INPUT_FILENAME)
        if not os.path.exists(input_path):
            skipped += 1
            report_rows.append(
                {
                    "folder": folder_name,
                    "folder_path": dirpath,
                    "input_csv": input_path,
                    "output_csv": "",
                    "agg_csv": "",
                    "status": "skipped_no_input_csv",
                    "error": "",
                }
            )
            continue

        output_name = f"{folder_name}_{INPUT_FILENAME}"
        output_path = os.path.join(dirpath, output_name)
        agg_path = os.path.join(AGG_DIR, output_name)

        try:
            df = pd.read_csv(input_path)
            cleaned_df, stats = clean_df(df)

            cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            cleaned_df.to_csv(agg_path, index=False, encoding="utf-8-sig")

            processed += 1
            report_rows.append(
                {
                    "folder": folder_name,
                    "folder_path": dirpath,
                    "input_csv": input_path,
                    "output_csv": output_path,
                    "agg_csv": agg_path,
                    "status": "ok",
                    "error": "",
                    **stats,
                }
            )
        except Exception as exc:
            errors += 1
            report_rows.append(
                {
                    "folder": folder_name,
                    "folder_path": dirpath,
                    "input_csv": input_path,
                    "output_csv": "",
                    "agg_csv": "",
                    "status": "error",
                    "error": str(exc),
                }
            )

    report_df = pd.DataFrame(report_rows)
    report_df.insert(0, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    print("====== Batch Cleaning Done (Stage3 + Aggregate) ======")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"AGG_DIR:  {AGG_DIR}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Report:    {REPORT_CSV}")


if __name__ == "__main__":
    main()
