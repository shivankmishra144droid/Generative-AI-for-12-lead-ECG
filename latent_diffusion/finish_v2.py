"""Finish Phase A audits after its worker exits. Never releases a failed gate."""
import argparse
import csv
import io
import json
import subprocess
import time
from .v2_common import ART,AUDITS,RESULTS,now,write_json


def exists(pid):
    result=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True,check=True)
    return any(len(row)>1 and row[1]==str(pid) for row in csv.reader(io.StringIO(result.stdout)))


def run(pid=None):
    started=now()
    while pid and exists(pid):
        if (ART/'STOP').exists():raise InterruptedError('STOP while awaiting phase A')
        time.sleep(20)
    completed=AUDITS/'phaseA.json'
    if not completed.exists() or not json.loads(completed.read_text()).get('completed'):
        raise RuntimeError('Phase A worker did not finish successfully; inspect its log before audit')
    from .finalize_phase_a_v2 import run as review
    from .final_audit_v2 import run as audit
    from .report_v2 import run as report
    review()
    result=audit()
    if not result['software_checks_pass']:raise RuntimeError('Software regression audit failed')
    report()
    write_json(RESULTS/'completion.json',dict(started_utc=started,finished_utc=now(),
        phaseA_work_complete=True,phaseA_gate_passed=result['experimental_gate_a_pass'],
        whole_v2_complete=False,remaining='B–G blocked by the original failed Gate A; see REPORT_V2.md'))
    print('Phase A implementation, rescoring, review and report complete. Gate A remains failed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    run(args.wait_pid)
