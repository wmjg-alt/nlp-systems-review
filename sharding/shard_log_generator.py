import json
import random
import time
from collections import defaultdict

# --- CONFIGURATION (TUNABLE KNOBS) ---
OUTPUT_LOG_FILE = "log.log"
OUTPUT_GROUND_TRUTH = "ground_truth.txt"

# Time Constants
SECONDS_IN_DAY = 86400
WINDOW_DAYS = 7

# Simulation Settings
TOTAL_LINES = 10000             # Size of the haystack
FAILURE_THRESHOLD = 50          # Count > 50 triggers a ban
NOISE_FAILURE_RATE = 0.02       # 2% of background traffic fails randomly

# Align Start Time to strict UTC Midnight.
# This prevents the "Bucket Split" bug where an attack spanning 
# 2PM-2PM gets split across two different UTC days by the // operator.
current_timestamp = int(time.time())
START_TIME = (current_timestamp // SECONDS_IN_DAY) * SECONDS_IN_DAY - (SECONDS_IN_DAY * WINDOW_DAYS)

# Known signals we are planting
# Format: (IP_Address, Day_Offset_0_to_6, Failure_Count)
PLANTED_BAD_ACTORS = [
    ("192.168.1.66", 2, FAILURE_THRESHOLD + 5),   # 55 fails (Should be BANNED)
    ("10.0.0.99",    5, FAILURE_THRESHOLD + 22),  # 72 fails (Should be BANNED)
    ("172.16.0.5",   1, FAILURE_THRESHOLD - 1),   # 49 fails (Edge case: SAFE)
]

def generate_random_ip():
    """Generates a random IPv4 address."""
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def main():
    print(f"--- CONFIGURATION ---")
    print(f"Start Time (UTC): {time.ctime(START_TIME)}")
    print(f"Threshold: > {FAILURE_THRESHOLD} failures/day")
    print(f"Noise Rate: {NOISE_FAILURE_RATE*100}%")
    print(f"---------------------")
    
    logs = []
    
    # --- STEP 1: PLANT THE SIGNAL ---
    print(f"🌱 Planting {len(PLANTED_BAD_ACTORS)} specific actors...")
    for ip, day_offset, count in PLANTED_BAD_ACTORS:
        # Ensure day_offset is within window
        if day_offset >= WINDOW_DAYS:
            print(f"⚠️ Warning: Actor {ip} offset {day_offset} is outside window.")
            continue

        day_start_ts = START_TIME + (day_offset * SECONDS_IN_DAY)
        
        for _ in range(count):
            # Random second within that specific UTC day
            offset = random.randint(0, SECONDS_IN_DAY - 1)
            ts = day_start_ts + offset
            
            logs.append({
                "ip_address": ip,
                "timestamp": ts,
                "status": "FAILURE",
                "meta": "planted_signal"
            })

    # --- STEP 2: FILL WITH NOISE ---
    remaining_lines = TOTAL_LINES - len(logs)
    print(f"📢 Generating {remaining_lines} lines of background noise...")
    
    for _ in range(remaining_lines):
        ip = generate_random_ip()
        
        # Random time anywhere in the 7 day window
        global_offset = random.randint(0, (SECONDS_IN_DAY * WINDOW_DAYS) - 1)
        ts = START_TIME + global_offset
        
        # Apply random failure rate
        is_failure = random.random() < NOISE_FAILURE_RATE
        status = "FAILURE" if is_failure else "SUCCESS"
        
        logs.append({
            "ip_address": ip,
            "timestamp": ts,
            "status": status
        })

    # --- STEP 3: SHUFFLE & WRITE ---
    print(f"🔀 Shuffling and writing to {OUTPUT_LOG_FILE}...")
    random.shuffle(logs)
    
    with open(OUTPUT_LOG_FILE, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

    # --- STEP 4: VERIFY GROUND TRUTH ---
    print(f"🔍 Verifying Ground Truth...")
    
    # Re-calculate truth from the actual list to catch any 
    # random noise that accidentally triggered a ban.
    
    # Structure: { IP: { DayInteger: Count } }
    validation_tracker = defaultdict(lambda: defaultdict(int))
    
    for entry in logs:
        if entry["status"] == "FAILURE":
            # integer division aligns strictly to UTC midnight buckets
            day_bucket = entry["timestamp"] // SECONDS_IN_DAY
            validation_tracker[entry["ip_address"]][day_bucket] += 1
            
    banned_ips = set()
    for ip, days in validation_tracker.items():
        for day, count in days.items():
            if count > FAILURE_THRESHOLD:
                banned_ips.add(ip)
                # If an IP is banned on Day 1, we don't need to check Day 2
                # (Assuming the list is just 'who is banned', not 'on which days')
                break 

    # --- STEP 5: SAVE TRUTH ---
    sorted_bans = sorted(list(banned_ips))
    print(f"✅ Found {len(sorted_bans)} banned IPs.")
    print(f"📝 Writing to {OUTPUT_GROUND_TRUTH}...")
    
    with open(OUTPUT_GROUND_TRUTH, "w") as f:
        for ip in sorted_bans:
            f.write(f"{ip}\n")
            
    print(f"\nEXPECTED BANS: {sorted_bans}")

if __name__ == "__main__":
    main()