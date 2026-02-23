import json
import sys
from pathlib import Path
from typing import Set, Dict, Any


def extract_subjects_comb(path: str) -> str:
    """
    Extract subjects_comb from a path.
    Format: cuboids_monochrome/subjects_comb/img_idx/cuboids.jpg
    """
    path_parts = path.split('/')
    
    if 'cuboids_monochrome' in path_parts:
        idx = path_parts.index('cuboids_monochrome')
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    
    return None


def get_subjects_combs_from_jsonl(jsonl_path: str) -> Set[str]:
    """Extract all unique subjects_comb from a JSONL file."""
    subjects_combs = set()
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                cv_path = entry.get('cv', '')
                subjects_comb = extract_subjects_comb(cv_path)
                
                if subjects_comb:
                    subjects_combs.add(subjects_comb)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in {jsonl_path}: {e}")
                continue
    
    return subjects_combs


def get_common_keys(jsonl_path1: str, jsonl_path2: str) -> Set[str]:
    """Get keys that are common across ALL entries in both JSONL files."""
    keys1 = None
    keys2 = None
    
    # Get keys from first file
    with open(jsonl_path1, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                current_keys = set(entry.keys())
                if keys1 is None:
                    keys1 = current_keys
                else:
                    keys1 = keys1.intersection(current_keys)
            except json.JSONDecodeError:
                continue
    
    # Get keys from second file
    with open(jsonl_path2, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                current_keys = set(entry.keys())
                if keys2 is None:
                    keys2 = current_keys
                else:
                    keys2 = keys2.intersection(current_keys)
            except json.JSONDecodeError:
                continue
    
    # Return intersection of keys from both files
    if keys1 is None or keys2 is None:
        return set()
    
    common_keys = keys1.intersection(keys2)
    return common_keys


def merge_jsonls(jsonl_path1: str, jsonl_path2: str, output_path: str):
    """
    Merge two JSONL files, ensuring:
    1. No overlapping subjects_comb (assertion)
    2. Only common keys are included in output
    3. All entries are concatenated
    """
    print(f"Checking for overlapping subjects_comb...")
    combs1 = get_subjects_combs_from_jsonl(jsonl_path1)
    combs2 = get_subjects_combs_from_jsonl(jsonl_path2)
    
    overlap = combs1.intersection(combs2)
    
    # Assert no overlap
    assert len(overlap) == 0, (
        f"ERROR: Found {len(overlap)} overlapping subjects_comb between files!\n"
        f"Overlapping subjects_comb: {sorted(overlap)}"
    )
    
    print(f"✓ No overlapping subjects_comb found")
    print(f"  File 1: {len(combs1)} unique subjects_comb")
    print(f"  File 2: {len(combs2)} unique subjects_comb")
    
    # Get common keys
    print(f"\nFinding common keys...")
    common_keys = get_common_keys(jsonl_path1, jsonl_path2)
    print(f"{common_keys = }")
    
    assert len(common_keys) > 0, "ERROR: No common keys found between the two JSONL files!"
    
    print(f"✓ Found {len(common_keys)} common keys: {sorted(common_keys)}")
    
    # Merge files
    print(f"\nMerging files to {output_path}...")
    total_entries = 0
    
    with open(output_path, 'w') as out_f:
        # Write entries from first file
        with open(jsonl_path1, 'r') as f1:
            for line_num, line in enumerate(f1, 1):
                try:
                    entry = json.loads(line.strip())
                    # Keep only common keys
                    filtered_entry = {k: entry[k] for k in common_keys if k in entry}
                    out_f.write(json.dumps(filtered_entry) + '\n')
                    total_entries += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num} in {jsonl_path1}: {e}")
                    continue
        
        # Write entries from second file
        with open(jsonl_path2, 'r') as f2:
            for line_num, line in enumerate(f2, 1):
                try:
                    entry = json.loads(line.strip())
                    # Keep only common keys
                    filtered_entry = {k: entry[k] for k in common_keys if k in entry}
                    out_f.write(json.dumps(filtered_entry) + '\n')
                    total_entries += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num} in {jsonl_path2}: {e}")
                    continue
    
    print(f"✓ Merged {total_entries} entries to {output_path}")
    print(f"\nMerge complete!")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_two_jsonls.py <jsonl_file1> <jsonl_file2> <output_jsonl>")
        print("\nExample:")
        print("  python merge_two_jsonls.py dataset1/cuboids.jsonl dataset2/cuboids.jsonl merged_cuboids.jsonl")
        sys.exit(1)
    
    jsonl_path1 = sys.argv[1]
    jsonl_path2 = sys.argv[2]
    output_path = sys.argv[3]
    
    # Validate input files
    if not Path(jsonl_path1).exists():
        print(f"Error: File not found: {jsonl_path1}")
        sys.exit(1)
    
    if not Path(jsonl_path2).exists():
        print(f"Error: File not found: {jsonl_path2}")
        sys.exit(1)
    
    # Check if output file already exists
    if Path(output_path).exists():
        response = input(f"Warning: {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Merge files
    try:
        merge_jsonls(jsonl_path1, jsonl_path2, output_path)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)