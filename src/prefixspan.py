from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Optional


Sequence  = List[int]          
Database  = List[Sequence]    
Pattern   = List[int]          
PatternDB = List[Tuple[Pattern, int]]   

def _project(database: Database, item: int) -> Database:
    projected: Database = []
    for seq in database:
        try:
            idx = seq.index(item)
        except ValueError:
            continue
        suffix = seq[idx + 1:]
        projected.append(suffix)
    return projected

def _frequent_items(database: Database, min_support: int) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for seq in database:
        for item in set(seq):         
            counts[item] += 1
    return {item: cnt for item, cnt in counts.items() if cnt >= min_support}


def _prefixspan_recursive(
    prefix:      Pattern,
    database:    Database,
    min_support: int,
    results:     PatternDB,
    max_len:     Optional[int],
) -> None:
    
    if max_len is not None and len(prefix) >= max_len:
        return

    freq_items = _frequent_items(database, min_support)

    for item, support in sorted(freq_items.items()):
        new_pattern  = prefix + [item]
        projected_db = _project(database, item)

        results.append((new_pattern, support))

        _prefixspan_recursive(
            prefix      = new_pattern,
            database    = projected_db,
            min_support = min_support,
            results     = results,
            max_len     = max_len,
        )


def prefixspan(
    database:    Database,
    min_support: int,
    max_len:     Optional[int] = None,
) -> PatternDB:
    
    results: PatternDB = []
    _prefixspan_recursive(
        prefix      = [],
        database    = database,
        min_support = min_support,
        results     = results,
        max_len     = max_len,
    )
    results.sort(key=lambda x: (-x[1], len(x[0])))
    return results



def mine_student_patterns(
    sequences:        List[List[str]],
    min_support:      int,
    max_len:          int = 4,
    item_to_id:       Optional[Dict[str, int]] = None,
) -> List[Tuple[List[str], int]]:
    

    if item_to_id is None:
        vocab     = sorted({item for seq in sequences for item in seq})
        item_to_id = {item: idx for idx, item in enumerate(vocab)}

    id_to_item = {v: k for k, v in item_to_id.items()}

    encoded: Database = [
        [item_to_id[item] for item in seq if item in item_to_id]
        for seq in sequences
    ]

    int_patterns = prefixspan(encoded, min_support=min_support, max_len=max_len)

    return [
        ([id_to_item[i] for i in pat], sup)
        for pat, sup in int_patterns
    ]

def discriminative_patterns(
    high_sequences: List[List[str]],
    low_sequences:  List[List[str]],
    min_support:    int,
    max_len:        int = 4,
    min_diff:       float = 0.05,
) -> List[Dict]:
    
    vocab = sorted(
        {item for seq in high_sequences + low_sequences for item in seq}
    )
    item_to_id = {item: idx for idx, item in enumerate(vocab)}

    high_patterns = dict(mine_student_patterns(high_sequences, min_support, max_len, item_to_id))
    low_patterns  = dict(mine_student_patterns(low_sequences,  min_support, max_len, item_to_id))

    n_high = max(len(high_sequences), 1)
    n_low  = max(len(low_sequences),  1)

    all_pats = set(map(tuple, high_patterns.keys())) | set(map(tuple, low_patterns.keys()))
    records  = []

    for pat_tuple in all_pats:
        pat       = list(pat_tuple)
        sup_high  = high_patterns.get(pat, 0)
        sup_low   = low_patterns.get(pat,  0)
        rel_high  = sup_high / n_high
        rel_low   = sup_low  / n_low
        diff      = abs(rel_high - rel_low)

        if diff >= min_diff:
            records.append({
                "pattern":      pat,
                "support_high": sup_high,
                "support_low":  sup_low,
                "rel_high":     round(rel_high, 4),
                "rel_low":      round(rel_low,  4),
                "diff":         round(diff,      4),
                "group":        "High" if rel_high > rel_low else "Low",
            })

    records.sort(key=lambda r: -r["diff"])
    return records
