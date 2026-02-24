import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# --- CONFIGURATION (TUNABLE KNOBS) ---
INPUT_FILE = "log.log"
OUTPUT_REPORT = "blacklist.txt"
TEMP_DIR = Path("temp_shards")

# Business Logic
FAILURE_THRESHOLD = 50        # Must match generator.py
SECONDS_IN_DAY = 86400
ESTIMATED_FAILURE_RATE = 0.05 # Conservative estimate (5%)

# --- SYSTEM CONSTRAINTS (THE INTERVIEW MAGIC) ---
# To test sharding on a tiny file, we pretend we only have 5KB of RAM.
# In a real interview/prod, you'd set this to: 64 * (1024**3)
GIGS_AVAILABLE = 64
SIMULATED_RAM_BYTES = 5 * 1024  

class SystemResourceError(Exception):
    pass

# --- HELPER: DYNAMIC SHARD CALCULATION ---
def calculate_shard_config(total_file_size):
    """
    Decides how many shards to create based on file size and available RAM.
    """
    # 1. How big is the data we actually care about? (Failures only)
    est_filtered_size = total_file_size * ESTIMATED_FAILURE_RATE
    
    # 2. We want a shard to fit comfortably in our RAM limit (e.g., 50% utilization)
    safe_shard_size = SIMULATED_RAM_BYTES * 0.5
    
    # 3. Calculate minimum shards needed
    # (avoid division by zero if safe_shard_size is tiny in testing)
    safe_shard_size = max(safe_shard_size, 1024) 
    
    min_shards = int(est_filtered_size // safe_shard_size) + 1
    
    # Force at least 2 shards for testing demonstration if file is small but > 0
    if total_file_size > 0 and min_shards < 2:
        min_shards = 2

    soft_limit = 1024
    max_safe_shards = soft_limit - 10
    
    if min_shards > max_safe_shards:
        raise SystemResourceError(
            f"Need {min_shards} shards but OS limit is {soft_limit}. Increase ulimit."
        )
        
    print(f"SYSTEM CONFIG:")
    print(f"   Input Size:      {total_file_size:,} bytes")
    print(f"   Simulated RAM:   {SIMULATED_RAM_BYTES:,} bytes")
    print(f"   Est. Fail Data:  {int(est_filtered_size):,} bytes")
    print(f"   Calculated Shards: {min_shards}")
    
    return min_shards

# --- HELPER: FAST PARSING ---
def extract_failure_record(line):
    """
    Returns dict if line is a FAILURE, else None.
    Optimized to avoid JSON parsing on Success logs.
    """
    if "FAILURE" not in line:
        return None
    try:
        record = json.loads(line)
        if record.get("status") == "FAILURE":
            return record
    except json.JSONDecodeError:
        pass
    return None

# --- PHASE 1: MAP (FILTER & SHARD) ---
def map_phase(input_path, num_shards):
    # Prepare Temp Directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir()
    
    shard_filenames = [TEMP_DIR / f"shard_{i}.jsonl" for i in range(num_shards)]
    file_handles = [open(f, 'w') for f in shard_filenames]
    
    print(f"  Phase 1: Filtering & Sharding...")
    try:
        with open(input_path, 'r') as infile:
            for line in infile:
                record = extract_failure_record(line)
                if record:
                    ip = record.get("ip_address", "unknown")
                    # CONSISTENT HASHING
                    shard_idx = hash(ip) % num_shards
                    file_handles[shard_idx].write(line)
    finally:
        for f in file_handles:
            f.close()
            
    return shard_filenames

# --- PHASE 2: REDUCE (AGGREGATE) ---
def reduce_phase(shard_paths):
    print(f"  Phase 2: Aggregating {len(shard_paths)} shards...")
    final_blacklist = set()
    
    for shard in shard_paths:
        # Optimization: Skip empty shards
        if shard.stat().st_size == 0:
            continue
            
        # Structure: { IP: { DayBucket: Count } }
        # This fits in RAM because the shard is small
        ip_stats = defaultdict(lambda: defaultdict(int))
        
        try:
            with open(shard, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    ip = data['ip_address']
                    ts = data['timestamp']
                    
                    # Align to Day Buckets (Integer Division)
                    day = ts // SECONDS_IN_DAY
                    ip_stats[ip][day] += 1
            
            # Check Thresholds
            for ip, days in ip_stats.items():
                for count in days.values():
                    if count > FAILURE_THRESHOLD:
                        final_blacklist.add(ip)
                        break
        finally:
            # CLEANUP: Delete shard immediately after processing to free disk
            shard.unlink()
            
    return final_blacklist

# --- MAIN ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"XX Error: {INPUT_FILE} not found. Run generator.py first.")
        return

    total_size = os.path.getsize(INPUT_FILE)
    
    # 1. Calculate Config
    try:
        num_shards = calculate_shard_config(total_size)
    except SystemResourceError as e:
        print(f"XX {e}")
        return

    # 2. Map
    shard_paths = map_phase(INPUT_FILE, num_shards)
    
    # 3. Reduce
    banned_ips = reduce_phase(shard_paths)
    
    # 4. Cleanup Temp Dir
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    # 5. Output
    print(f"✅ Found {len(banned_ips)} banned IPs.")
    print(f"📝 Writing to {OUTPUT_REPORT}...")
    
    sorted_ips = sorted(list(banned_ips))
    with open(OUTPUT_REPORT, 'w') as f:
        for ip in sorted_ips:
            f.write(f"{ip}\n")
            
    print(f"BANNED IPS: {sorted_ips}")

if __name__ == "__main__":
    main()