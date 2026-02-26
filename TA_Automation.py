import os
import time
import json
import numpy as np
import requests
import pandas as pd
import gspread
from datetime import datetime
from zoneinfo import ZoneInfo
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from concurrent.futures import ThreadPoolExecutor

# -------------------- START TIMER --------------------
start_time = time.time()

# -------------------- ENV & AUTH --------------------
sec = os.getenv("ASHRITHA_SECRET_KEY")
User_name = os.getenv("USERNAME")
service_account_json = os.getenv("SERVICE_ACCOUNT_JSON")
MB_URL = os.getenv("METABASE_URL").rstrip("/") + "/api/session"
SAK = os.getenv("TA_SHEET_ACCESS_KEY")
TA_SHEET_KEY = os.getenv("TA_SHEET_ACCESS_KEY")  # ✅ FIX #1: was missing, caused NameError

# -------------------- QUERY ENV VARS --------------------
TA_SESSIONS_QUERY = os.getenv("TA_SESSIONS_QUERY")   # card/6135
TA_BATCH_QUERY    = os.getenv("TA_BATCH_QUERY")       # card/6570
TA_SLOTS_QUERY    = os.getenv("TA_SLOTS_QUERY")       # card/6213

if not sec or not service_account_json:
    raise ValueError("❌ Missing environment variables. Check GitHub secrets.")

if not TA_SHEET_KEY:
    raise ValueError("❌ TA_SHEET_ACCESS_KEY is not set. Check GitHub secrets.")

# -------------------- GOOGLE AUTH --------------------
service_info = json.loads(service_account_json)
creds = Credentials.from_service_account_info(
    service_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
)
gc = gspread.authorize(creds)

# -------------------- METABASE AUTH --------------------
res = requests.post(
    MB_URL,
    headers={"Content-Type": "application/json"},
    json={"username": User_name, "password": sec}
)

print("Status code:", res.status_code)
print("Response text:", res.text)  # ← add this

res.raise_for_status()
token = res.json()['id']
METABASE_HEADERS = {
    'Content-Type': 'application/json',
    'X-Metabase-Session': token
}
print("✅ Metabase session created")

# -------------------- UTILITIES --------------------
def fetch_with_retry(url, headers, retries=5, delay=15):
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"[Metabase] Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise

def safe_clear_and_update(worksheet, df):
    """Clears sheet and writes fresh data with retry logic."""
    title = worksheet.title
    print(f"🔄 Updating sheet: {title}")
    for attempt in range(1, 6):
        try:
            worksheet.clear()
            set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)
            print(f"✅ Successfully updated: {title}")
            return
        except Exception as e:
            print(f"[Sheets] Attempt {attempt} failed for {title}: {e}")
            if attempt < 5:
                time.sleep(20)
            else:
                print(f"❌ All attempts failed for {title}.")
                raise

# -------------------- FETCH ALL 3 QUERIES IN PARALLEL --------------------
print("📡 Fetching TA Sessions, Batch, and Slots data in parallel...")

urls = {
    "sessions": TA_SESSIONS_QUERY,
    "batch":    TA_BATCH_QUERY,
    "slots":    TA_SLOTS_QUERY
}

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {name: executor.submit(fetch_with_retry, url, METABASE_HEADERS) for name, url in urls.items()}
    results = {name: f.result() for name, f in futures.items()}

print("✅ All queries fetched successfully")

# -------------------- PROCESS df1 (Sessions Feedback) --------------------
df1 = pd.DataFrame(results["sessions"].json())

# ✅ FIX #2: Added 'session_start_time' to column selection — it's needed for feature engineering
df1 = df1[[
    'subjective_feedback', 'lu_batch_name', 'au_batch_name', 'au_start_date',
    'feedback_given', 'session_id', 'rating', 'description', 'module_name',
    'topic', 'cancel_reason', 'action_time', 'booked_time', 'session_start_time'
]]
df1 = df1.rename(columns={'lu_batch_name': 'Batch', 'module_name': 'Module'})

# -------------------- PROCESS df2 (Batch Info) --------------------
df2 = pd.DataFrame(results["batch"].json())
df2 = df2.rename(columns={'batch': 'Batch'})

# ✅ FIX #3: Verify required merge columns exist in df2 before proceeding
required_df2_cols = {'session_id', 'Batch', 'mentor_name', 'time_category', 'session_start_time'}
missing = required_df2_cols - set(df2.columns)
if missing:
    raise ValueError(f"❌ df2 (batch query) is missing expected columns: {missing}")

# -------------------- MERGE df1 + df2 --------------------
df = pd.merge(df1, df2, on=['session_id', 'Batch'], how='inner')

# -------------------- FEATURE ENGINEERING --------------------
df['session_start_time'] = pd.to_datetime(df['session_start_time'], errors='coerce')
df['au_start_date']      = pd.to_datetime(df['au_start_date'], errors='coerce')

df['month_diff_period'] = (
    df['session_start_time'].dt.to_period('M').astype(int) -
    df['au_start_date'].dt.to_period('M').astype(int)
)

# ✅ FIX #4: Safer rating deduplication — uses positional index within group, not global DataFrame index
df['rating'] = df.groupby('session_id')['rating'].transform(
    lambda x: x.mask(np.arange(len(x)) != 0, np.nan)
)

df['year_month_date_hour'] = df['session_start_time'].dt.strftime('%Y-%m-%d-%H')
df = df.drop_duplicates()

# -------------------- PROCESS df3 (TA Slots) --------------------
df3 = pd.DataFrame(results["slots"].json())
df3['date'] = pd.to_datetime(df3['date'])
df3['year_month_date_hour'] = df3['date'].dt.strftime('%Y-%m-%d-%H')
df3 = df3.rename(columns={'ta': 'mentor_name'})

# -------------------- MERGE df + df3 → df4 --------------------
df4 = pd.merge(df, df3, on=['year_month_date_hour', 'mentor_name', 'time_category'], how='outer')

# -------------------- WRITE TO GOOGLE SHEETS --------------------
print("📝 Connecting to Google Sheets...")
sheet = gc.open_by_key(TA_SHEET_KEY)  # ✅ FIX #1: TA_SHEET_KEY is now properly defined

safe_clear_and_update(sheet.worksheet("TA_sessions"),     df)
time.sleep(3)
safe_clear_and_update(sheet.worksheet("TA_slots"),        df3)
time.sleep(3)
safe_clear_and_update(sheet.worksheet("TA-sessions-all"), df4)

# -------------------- TIMESTAMP --------------------
current_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%b-%Y %H:%M:%S")
# Optional: uncomment if you have a dedicated timestamp cell
# ws_pivot = sheet.worksheet("TA-sessions-all")
# ws_pivot.update("B1", [[current_time]])
print(f"✅ Timestamp: {current_time}")

# -------------------- TIMER --------------------
end_time = time.time()
mins, secs = divmod(end_time - start_time, 60)
print(f"⏱ Total time: {int(mins)}m {int(secs)}s")
print("🎯 TA Sessions + Slots pipeline completed successfully!")
