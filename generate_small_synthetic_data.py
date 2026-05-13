#!/usr/bin/env python3
"""
Small Synthetic Dataset Generator for SOAPIE SLM - AFRCC Case Notes Quality Classification
Generates <=100 realistic case notes labeled "good" vs "incomplete" following DHA "Writing Case Notes" training.
Aligned to AFRCC: accurate/timely documentation, trauma-informed, measurable goals, no unapproved jargon.
Total: 80 examples (40 good, 40 incomplete)
Splits: Train 56 (70%), Validation 12 (15%), Test 12 (15%)
"""

import json
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

random.seed(2026)  # Reproducible

OUTPUT_DIR = "/home/workdir/artifacts/soapie_slm_small_afrcc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vocab
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie", "Avery", "Quinn", "Skyler"]
LAST_NAMES = ["Rivera", "Chen", "Patel", "Kim", "Nguyen", "O'Brien", "Santos", "Thompson", "Garcia", "Washington"]
RANKS = ["SSgt", "TSgt", "MSgt", "Capt", "Maj", "SrA", "A1C", "Amn", "2nd Lt"]
BASES = ["JBSA-Lackland", "JB Andrews", "Travis AFB", "Eglin AFB", "Nellis AFB", "Barksdale AFB"]
CONDITIONS = [
    "combat-related mild TBI", "PTSD with depression", "right leg amputation (below knee)",
    "severe burn injuries (25% BSA)", "chronic back pain post-spinal fusion",
    "post-concussion syndrome", "adjustment disorder with anxiety", "polytrauma (fractures + TBI)",
    "military sexual trauma recovery", "caregiver burnout (spouse of SM w/ TBI)", "rehab post-rotator cuff"
]
INTERACTION_TYPES = ["in-person at MTF", "video telehealth", "phone follow-up (audio-only)", "secure email survey response"]

def generate_sm_id():
    return f"SM-{random.randint(100000, 999999)}"

def generate_date(days_ago=0):
    base = datetime(2025, 10, 20)
    return (base - timedelta(days=days_ago + random.randint(0, 20))).strftime("%Y-%m-%d")

def generate_good_case_note() -> Tuple[str, Dict[str, Any]]:
    """Generate a HIGH-QUALITY case note following all SOAPIE best practices."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    rank = random.choice(RANKS)
    base = random.choice(BASES)
    sm_id = generate_sm_id()
    cond = random.choice(CONDITIONS)
    date = generate_date()
    itype = random.choice(INTERACTION_TYPES)
    
    # Realistic good note: all sections present, concise, measurable goals, trauma-informed, no jargon or explained
    raw = f"""Case Note - {itype.upper()} | {date} | {sm_id} | {rank} {name} | {base}
Condition: {cond} | Caregiver present: {'Yes' if random.random() > 0.5 else 'No'} | Designated Caregiver: Spouse

SUBJECTIVE: Service member reports "PT is really helping, pain down to 3/10 from 8/10 last month. Sleeping through the night for first time in years. The support group at the MTF helped a lot - I don't feel so alone." Caregiver notes: "My husband is engaging with the kids more, laughing again. He's back to light duty at work part-time."

OBJECTIVE: RCC observed Service member maintained good eye contact, smiled when discussing family, walked with improved gait without cane today. Client appeared well-groomed, engaged actively in goal-setting conversation, no visible signs of distress or guarding. During video call, Service member was in uniform (light duty), background showed family photos, spoke clearly and confidently.

ASSESSMENT: Service member demonstrates clear progress in physical and emotional domains. Recognizes value of multidisciplinary approach (PT + counseling + peer support). Caregiver reports improved family engagement, indicating reduced isolation risk. No acute safety concerns; motivated for continued recovery.

PLAN: 
1. Continue current PT 3x/week; reassess pain scale in 2 weeks (target: maintain <4/10).
2. Attend next 2 support group sessions; caregiver to join 1 virtual family session by 11/15/2025.
3. Complete college application draft by 11/01/2025 (measurable milestone toward education goal).
4. Schedule follow-up RCC check-in via video on 11/05/2025 at 1400.

IMPLEMENTATION: As planned from prior note (10/05/2025), Service member and spouse attended military job fair at Military & Family Readiness Center on 10/12/2025. Service member submitted 2 job applications and spoke with 3 employers. Also completed 4/6 PT sessions this period and attended 1 support group meeting.

EVALUATION: Service member received tentative job offer (pending background check) as direct result of job fair attendance - strong outcome vs goal. Pain management and sleep improvements sustained; family reports "more hopeful." Progress on track. Recommend step-down to bi-weekly RCC contact if gains maintained over next 3 weeks. No unaddressed barriers identified.

RCC: Jordan Rivera, Recovery Care Coordinator | Entered: {date} 1620 | DoD-CMS Case #{random.randint(10000,99999)} | HIPAA compliant - minimal necessary info only."""

    metadata = {
        "label": "good",
        "quality_score": random.randint(82, 96),
        "soapie_completeness": 1.0,
        "issues": [],
        "measurable_goals_present": True,
        "trauma_informed": True,
        "no_unapproved_jargon": True,
        "concise": True,
        "timely": True
    }
    return raw, metadata

def generate_incomplete_case_note() -> Tuple[str, Dict[str, Any]]:
    """Generate an INCOMPLETE/POOR case note violating guidelines (missing sections, jargon, no metrics, verbose, etc.)."""
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    rank = random.choice(RANKS)
    base = random.choice(BASES)
    sm_id = generate_sm_id()
    cond = random.choice(CONDITIONS)
    date = generate_date()
    itype = random.choice(INTERACTION_TYPES)
    
    error_type = random.choice(["missing_sections", "jargon_heavy", "no_measurable_goals", "verbose_subjective_only", "grammar_errors", "untimely_vague"])
    
    if error_type == "missing_sections":
        raw = f"""Quick note on {sm_id} {rank} {name} at {base} re: {cond}. 
Spoke today - says pain is bad, can't sleep, snapping at family. Caregiver worried about flashbacks coming back. 
Told him to keep doing PT and call if worse. Will check in later. 
SM is OEF vet, 3 deployments, on MEB for this. 
[Note entered 3 days late - was busy with other cases]"""
        issues = ["missing objective observations", "missing assessment (professional opinion)", "missing plan with timelines", "missing implementation/evaluation", "uses unexplained acronym 'OEF' 'MEB'", "vague 'call if worse' no timeline", "entered late (not timely)"]
        quality = random.randint(28, 48)
        
    elif error_type == "jargon_heavy":
        raw = f"""Case Note - {itype} | {date}
SM {sm_id} {rank} {name} {base} dx: {cond} s/p OIF/OEF, PMH: PTSD, MDD, chronic LBP s/p L4-5 fusion. 
CC: c/o 9/10 LBP radiating to RLE, +insomnia, +anhedonia, +hypervigilance. 
Caregiver: spouse reports SM "not himself", irritable, isolative. 
O/E: A&Ox3, mood dysphoric, affect constricted, no SI/HI. 
A/P: Continue current meds (sertraline 100mg, gabapentin 600mg TID, percocet PRN), f/u PT, refer to BH for med mgmt. RTC 4 wks or PRN. 
RCC: Capt Kim | {date}"""
        issues = ["heavy unexplained military/medical jargon (OIF/OEF, PMH, s/p, c/o, O/E, A&Ox3, SI/HI, RTC, PRN, TID, dx, MDD, LBP)", "no full SOAPIE structure", "plan not measurable (no specific goals/timelines for SM)", "assessment too brief, no holistic view", "abbreviations not approved per Service guidelines"]
        quality = random.randint(35, 55)
        
    elif error_type == "no_measurable_goals":
        raw = f"""Follow up with {rank} {name} ({sm_id}) today via {itype}. Condition {cond}. 
Service member says he's trying but it's hard. Pain is still there but better some days. 
Talked about family stuff - wife is stressed with the kids and appointments. 
I encouraged him to keep going to PT and counseling. He seemed okay with that. 
We'll see how he does and touch base again soon. Caregiver was on the call too and agrees.
No new issues. 
RCC note entered same day."""
        issues = ["plan lacks measurable goals or timelines (no deadlines, no specific actions with dates)", "subjective heavy, minimal objective data", "assessment vague ('seemed okay')", "no implementation details from prior plan", "evaluation missing (no outcome vs goals)", "no specific recommendation or follow-up date"]
        quality = random.randint(40, 58)
        
    elif error_type == "verbose_subjective_only":
        raw = f"""Long interaction today with {rank} {name} {sm_id} at {base} for his {cond}. He talked for almost 45 minutes straight about everything going on - the pain, how the flashbacks are worse at night, how he feels guilty about not being able to play with his kids like before the injury, how his wife is carrying everything and he feels like a burden, how the medications make him foggy but without them the pain is unbearable, how he misses his battle buddies and feels like he's letting everyone down, the upcoming MEB process is stressing him out because he doesn't know what the future holds for his career or benefits, and on and on. I listened and validated his feelings as best I could. He cried a bit when talking about the kids. Caregiver called in at the end and said she's worried he's not eating or sleeping well. I told them both that this is normal in recovery and to reach out to the chaplain or BH if it gets too heavy. We went over his current meds and PT schedule. I think he just needed to vent today. Will document in DoD-CMS later tonight after I finish notes for 4 other SMs. This one took a lot out of me emotionally too."""
        issues = ["excessively verbose (over 250 words, violates 'concise' guideline)", "almost entirely subjective - no structured objective observations or RCC analysis", "no clear Plan with actions/goals", "no Implementation or Evaluation sections", "emotional boundary issue (RCC personalizing 'took a lot out of me')", "no measurable elements", "privacy risk: overly detailed narrative without focus"]
        quality = random.randint(25, 45)
        
    elif error_type == "grammar_errors":
        raw = f"""case note for {sm_id} {rank} {name} {base} {cond} {itype} on {date}
sm said his pain is worst then b4 and he cant sleep good no more. wife says hes snappy with the kids and not eating rite. i told him to do his pt exercises and take his pills. he looked tired and sad. caregiver on phone too. we talked about setting some goals but didnt pick dates. will f/u next week sometime. sm is army vet from iraq war with tbi and ptsd. no si. 
entered by rcc {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"""
        issues = ["grammar/spelling errors ('worst then b4', 'cant sleep good no more', 'snappy', 'rite', 'f/u', 'b4')", "inconsistent capitalization and punctuation", "mixes military jargon without definition", "no SOAPIE structure or clear sections", "plan vague ('next week sometime' - not measurable)", "incomplete sentences", "poor readability per guideline"]
        quality = random.randint(22, 42)
        
    else:  # untimely_vague
        raw = f"""Old note: Spoke with {rank} {name} about his recovery from {cond}. Things are okay I guess. He mentioned some stress at home. I advised him to talk to his wife and keep up with appointments. No big changes. Will monitor. 
[This note was written 11 days after the actual phone call on {generate_date(11)} because I was TDY and then had leave. Sorry for delay.]"""
        issues = ["entered 11 days late (violates 'timely' and 'immediate documentation' guideline)", "extremely vague ('things are okay I guess', 'some stress')", "no SOAPIE elements at all", "no objective observations", "no plan, implementation, or evaluation", "personal apology in note (unprofessional, belongs in separate admin note)", "no measurable goals or follow-up specifics"]
        quality = random.randint(18, 38)
    
    metadata = {
        "label": "incomplete",
        "quality_score": quality,
        "soapie_completeness": round(random.uniform(0.2, 0.6), 1),
        "issues": issues,
        "measurable_goals_present": False,
        "trauma_informed": random.choice([True, False]),
        "no_unapproved_jargon": False,
        "concise": False,
        "timely": error_type != "untimely_vague"
    }
    return raw, metadata

def main():
    all_examples = []
    for i in range(40):
        raw, meta = generate_good_case_note()
        ex = {
            "id": f"AFRCC-SOAPIE-GOOD-{1000+i}",
            "raw_case_note": raw,
            "label": meta["label"],
            "quality_score": meta["quality_score"],
            "soapie_completeness": meta["soapie_completeness"],
            "issues": meta["issues"],
            "condition": random.choice(CONDITIONS),
            "interaction_type": random.choice(INTERACTION_TYPES),
            "date": generate_date(),
            "metadata": meta
        }
        all_examples.append(ex)
    
    for i in range(40):
        raw, meta = generate_incomplete_case_note()
        ex = {
            "id": f"AFRCC-SOAPIE-INCOMP-{2000+i}",
            "raw_case_note": raw,
            "label": meta["label"],
            "quality_score": meta["quality_score"],
            "soapie_completeness": meta["soapie_completeness"],
            "issues": meta["issues"],
            "condition": random.choice(CONDITIONS),
            "interaction_type": random.choice(INTERACTION_TYPES),
            "date": generate_date(),
            "metadata": meta
        }
        all_examples.append(ex)
    
    random.shuffle(all_examples)
    
    # Splits 70/15/15 = 56 / 12 / 12
    train = all_examples[:56]
    val = all_examples[56:68]
    test = all_examples[68:80]
    
    def save_jsonl(data, name):
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} to {name}")
    
    save_jsonl(train, "train.jsonl")
    save_jsonl(val, "validation.jsonl")
    save_jsonl(test, "test.jsonl")
    
    # Also save a small CSV preview for easy import
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "sample_preview.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "quality_score", "soapie_completeness", "condition", "raw_preview"])
        for ex in all_examples[:20]:  # first 20 for preview
            preview = ex["raw_case_note"][:180].replace("\n", " ") + "..."
            writer.writerow([ex["id"], ex["label"], ex["quality_score"], ex["soapie_completeness"], ex["condition"][:30], preview])
    
    print(f"\nTotal generated: {len(all_examples)} (40 good, 40 incomplete)")
    print(f"Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    print(f"Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()