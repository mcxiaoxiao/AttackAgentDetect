"""Fetch pinned KYPO human logs and export source/variant/level task traces."""
import collections
import csv
import datetime
import hashlib
import io
import json
import pathlib
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from attackagent.traces import ATOM_GROUP, command_heads

RAW = ROOT / 'research/data/external/kypo-human-v1'
OUT = ROOT / 'research/results/human_task_supplement_v1'
SITE = ROOT / 'site/data'
PINS = {'data1.zip':'ceae9c1dd60986262e5f9e6456fa75266a98f134b2032b6ef32275d596699adc',
        'data2.zip':'e2099c2ac919f534f23c89f4f34c01c7426c81a95e682b5b5349c2d8e11dec85'}
URL = 'https://zenodo.org/records/10298187'


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    records, registry, archives, training_definitions = [], [], [], {}
    participants = set(); raw_counts = collections.Counter()
    for archive_name, digest in PINS.items():
        p = RAW / archive_name
        if not p.exists():
            with urllib.request.urlopen(URL+'/files/'+archive_name+'?download=1', timeout=40) as response:
                p.write_bytes(response.read())
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != digest: raise ValueError('Pinned archive hash mismatch: '+archive_name)
        variant = p.stem
        archives.append({'file':archive_name,'sha256':digest,'bytes':p.stat().st_size,'url':URL+'/files/'+archive_name+'?download=1'})
        with zipfile.ZipFile(p) as z:
            definition = json.loads(z.read(variant+'/training_definition.json'))
            training_definitions[variant] = definition
            events = list(csv.DictReader(io.StringIO(z.read(variant+'/process_mining.csv').decode())))
        grouped = collections.defaultdict(list)
        for row in events:
            participants.add((variant,row['sandbox_id']))
            raw_counts[row['cmd_type']] += 1
            if row['cmd_type'] not in ('bash-command','msf-command'): continue
            grouped[(row['sandbox_id'],row['level'])].append(row)
        for (sandbox, level), rows in sorted(grouped.items()):
            rows.sort(key=lambda x: datetime.datetime.fromisoformat(x['timestamp_str'].replace('Z','+00:00')))
            commands = [c for row in rows if row['cmd_type']=='bash-command' for c in command_heads(row['cmd_name'])]
            task = 'kypo-locust/'+variant+'/level-'+level
            participant = hashlib.sha256((variant+'/'+sandbox).encode()).hexdigest()[:16]
            trace = hashlib.sha256((task+'/'+participant).encode()).hexdigest()[:20]
            registry.append({'trace_id':trace,'source_dataset':'kypo-locust','variant':variant,'task_group':task,
                'participant_id':participant,'mode':'human','observed_command_events':len(rows),
                'common_command_count':len(commands),'source_record':variant+'/process_mining.csv#sandbox='+sandbox+';level='+level,
                'license':'CC-BY-4.0','relation_scope':'same-source-variant-level; no matched Agent runs'})
            records.append({'trace_id':trace,'commands':commands,'action_counts':dict(collections.Counter(ATOM_GROUP[c] for c in commands))})
    coverage=[]
    for minimum in (1,4,8,16,32):
        eligible=[x for x in registry if x['common_command_count']>=minimum]
        counts=collections.Counter(x['task_group'] for x in eligible)
        repeated={t for t,n in counts.items() if n>=2}
        coverage.append({'minimum_commands':minimum,'records':len(eligible),'tasks':len(counts),'tasks_with_2_humans':len(repeated),
            'records_in_repeated_tasks':sum(x['task_group'] in repeated for x in eligible),
            'participants':len({x['participant_id'] for x in eligible})})
    summary={'date':'2026-11-02','source':URL,'license':'CC-BY-4.0','participants':len(participants),
        'task_level_segments':len(registry),'nonempty_common_view_segments':sum(x['common_command_count']>0 for x in registry),
        'source_variant_levels':len({x['task_group'] for x in registry}),'raw_event_counts':dict(raw_counts),'length_coverage':coverage,
        'human_agent_matched_tasks':0,'status':'ingested_human_only_task_level_supplement_not_evaluated',
        'limits':['Segments from the same participant are dependent. Split by participant before windows/pairs.',
                  'Data1 and data2 retain separate task namespaces; no cross-variant equivalence assumed.',
                  'Author-provided level labels delimit training tasks; not real attack campaign identities.',
                  'No Agent runs in this source; do not pool source labels into a Human/Agent classifier.',
                  'Common view retains known Bash/ZSH command heads only. Metasploit events remain in raw archive, excluded from common shell view.']}
    write_csv(OUT/'trace_registry.csv',registry)
    (OUT/'evidence.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'training_definitions.json').write_text(json.dumps(training_definitions,indent=2)+'\n')
    manifest={'source':URL,'license':'CC-BY-4.0','archives':archives,'normalization':'known-command-heads-lossy-v1; Bash cmd_name only; excludes game and Metasploit events'}
    (ROOT/'research/data/manifests/human_task_kypo_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps({**manifest,'builder_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'crosswalk_source':'research/results/behavior_corpus_v1/trace_registry.csv','crosswalk_method':'exact normalized challenge slug only; no fixture equivalence asserted'},indent=2)+'\n')
    write_csv(SITE/'human-task-supplement-v1.csv',registry)
    (SITE/'human-task-supplement-v1.json').write_text(json.dumps(summary,indent=2)+'\n')
    base=ROOT/'research/results/behavior_corpus_v1'
    corpus_meta=list(csv.DictReader((base/'trace_registry.csv').open()))
    evidence={}
    for line in (base/'evidence.jsonl').open():
        item=json.loads(line);evidence[item['trace_id']]=item
    humans=collections.defaultdict(list);agents=collections.defaultdict(list)
    for item in corpus_meta:
        key=item['task_id']
        if item['source_dataset']=='cyber-task-horizons' and item['benchmark']=='cybench' and item['mode']=='human':
            humans[key].append(item)
        elif item['source_dataset']=='antieval/cybench-trajectories':
            key=key.removeprefix('cybench:').removesuffix(' (hard)')
            agents[key].append(item)
    crosswalk=[]
    for task in sorted(set(humans)&set(agents)):
        hcounts=[len(evidence[x['trace_id']]['commands']) for x in humans[task]]
        acounts=[len(evidence[x['trace_id']]['commands']) for x in agents[task]]
        crosswalk.append({'challenge_id':task,'human_source':'cyber-task-horizons/cybench','agent_source':'antieval/cybench-trajectories/cybench-hard',
            'human_traces':len(hcounts),'human_traces_ge4':sum(n>=4 for n in hcounts),'human_traces_ge8':sum(n>=8 for n in hcounts),
            'agent_traces':len(acounts),'agent_traces_ge4':sum(n>=4 for n in acounts),'agent_traces_ge8':sum(n>=8 for n in acounts),
            'relation_status':'same_challenge_slug; fixture_version_and_capture_parity_unverified'})
    write_csv(SITE/'human-agent-cybench-crosswalk-v1.csv',crosswalk)
    write_csv(OUT/'human-agent-cybench-crosswalk-v1.csv',crosswalk)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
