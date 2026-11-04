"""Optional W&B export of saved evidence. Never generates or selects edits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def snapshots(folder):
    """One tracking run per hosted attempt, with separate metric namespaces."""
    folder=Path(folder)
    plan,state=read(folder/"plan.json"),read(folder/"state.json")
    for child in sorted(folder.glob("round-*")):
        if not (child/"submission.json").exists():
            continue
        submission=read(child/"submission.json")
        report=read(child/"verifier-report.json") if (child/"verifier-report.json").exists() else None
        failure=read(child/"execution-failure.json") if (child/"execution-failure.json").exists() else None
        if report is None and failure is None:
            continue  # A running attempt is not a performance measurement.
        eid=submission["evaluation_id"]
        run_id=hashlib.sha256((plan["model_repo"]+"/"+eid).encode()).hexdigest()[:16]
        metrics={"trial/round":submission["round_index"]+1,
                 "execution/failed":int(report is None),"selection/accepted":int(bool(report and report["accepted"]))}
        if report:
            metrics["candidate/valid"]=int(report["valid"])
            if "changed_parameters" in report:
                metrics["candidate/changed_parameters"]=report["changed_parameters"]
            for section,prefix in (("practice_qualification","qualification"),("gate","gate"),("final_audit","audit")):
                comparison=report.get(section,{})
                for key,value in comparison.items():
                    if type(value) in (int,float):
                        metrics[prefix+"/"+key]=value
                if comparison.get("baseline_returns") and comparison.get("candidate_returns"):
                    delta=[b-a for a,b in zip(comparison["baseline_returns"],comparison["candidate_returns"],strict=True)]
                    metrics.update({prefix+"/wins":sum(d>0 for d in delta),prefix+"/losses":sum(d<0 for d in delta),
                                    prefix+"/ties":sum(d==0 for d in delta),prefix+"/worst_paired_change":min(delta)})
            selected_hash=report["selected_hash"]
        else:
            selected_hash=submission["baseline_hash"]
        practice=[]
        if (child/"trace.json").exists():
            for node in read(child/"trace.json").get("nodes",[]):
                m=node["message"]
                if m.get("role")!="tool" or m.get("name") not in ("repair_test_edits","repair_propose_edits"):
                    continue
                try: result=json.loads(m["content"])
                except (ValueError,TypeError): continue
                if not isinstance(result,dict) or "candidate_mean" not in result or result.get("valid") is False:
                    continue
                row={"practice/proposal":len(practice)+1}
                for key in ("baseline_mean","candidate_mean","practice_gain","practice_paired_standard_error","changed_parameters"):
                    if type(result.get(key)) in (int,float): row["practice/"+key]=result[key]
                practice.append(row)
        publication=read(child/"publication.json") if (child/"publication.json").exists() else {}
        if (child/"deployment-check.json").exists(): metrics["deployment/verified"]=1
        files=[folder/"plan.json"]
        for name in ("baseline-policy.json","candidate-policy.json","verifier-report.json","submission.json",
                     "publication.json","deployment-check.json","execution-failure.json"):
            if (child/name).exists(): files.append(child/name)
        digest=hashlib.sha256()
        for path in files: digest.update(path.name.encode()+path.read_bytes())
        for row in practice: digest.update(json.dumps(row,sort_keys=True).encode())
        digest.update(json.dumps(metrics,sort_keys=True).encode())
        yield {"id":run_id,"name":folder.name+"-"+child.name,"plan":plan,"metrics":metrics,
               "practice":practice,"files":files,"report":report,"failure":failure,
               "evaluation_id":eid,"selected_hash":selected_hash,"publication":publication,
               "data_sha256":digest.hexdigest(),"child":child}


def sync(folder, entity=None, project=None):
    import wandb
    entity=entity or os.environ.get("WANDB_ENTITY") or wandb.Api(timeout=30).default_entity
    project=project or os.environ.get("WANDB_PROJECT") or "weight-space-breakout"
    folder=Path(folder)
    urls=[]
    for item in snapshots(folder):
        plan=item["plan"]
        protocol=plan.get("protocol_id","one-shot-iterative")
        run=wandb.init(entity=entity,project=project,id=item["id"],resume="allow",
            name=item["name"],group=folder.name,job_type="policy-repair-evaluation",
            tags=[protocol,"execution-failure" if item["failure"] and not item["report"] else "verified-evaluation"],
            config={"protocol":protocol,"environment":plan["environment"],"editor":plan["editor"],
                    "temperature":plan["temperature"],"model_repo":plan["model_repo"],
                    "promotion_rule":plan["promotion_rule"],"evaluation_id":item["evaluation_id"]},
            dir=str(folder),save_code=False,
            settings=wandb.Settings(base_url="https://api.wandb.ai",console="off",disable_git=True,
                disable_code=True,x_disable_stats=True,x_disable_meta=True,x_disable_machine_info=True,init_timeout=45))
        try:
            if run.summary.get("data_sha256") != item["data_sha256"]:
                if not run.summary.get("metrics_logged",False):
                    run.define_metric("practice/proposal")
                    run.define_metric("practice/*",step_metric="practice/proposal")
                    for row in item["practice"]: run.log(row)
                    run.log(item["metrics"])
                    run.summary["metrics_logged"]=True
                run.summary.update(item["metrics"])
                run.summary.update({"data_sha256":item["data_sha256"],"selected_policy_hash":item["selected_hash"],
                    "prime_url":None if plan.get("compute_platform")=="huggingface-jobs" else "https://app.primeintellect.ai/dashboard/evaluations/"+item["evaluation_id"],
                    "execution_url":plan.get("job_url") or "https://app.primeintellect.ai/dashboard/evaluations/"+item["evaluation_id"],
                    "hf_model_url":"https://huggingface.co/"+plan["model_repo"],
                    "hf_revision":item["publication"].get("revision"),
                    "interpretation":"Practice is exploratory. Gate controls promotion. Audit never selects edits. Execution failures have no policy-performance score."})
                evidence=wandb.Artifact("evidence-"+item["id"],type="evaluation",metadata={"data_sha256":item["data_sha256"]})
                for path in item["files"]: evidence.add_file(str(path),name=path.name)
                run.log_artifact(evidence)
                selected=item["child"]/("candidate-policy.json" if item["report"] and item["report"]["accepted"] else "baseline-policy.json")
                if selected.exists():
                    weights=wandb.Artifact("policy-"+item["id"],type="model",metadata={"policy_hash":item["selected_hash"],"hf_revision":item["publication"].get("revision")})
                    weights.add_file(str(selected),name="selected-policy.json")
                    run.log_artifact(weights)
            urls.append(run.url)
        finally:
            run.finish(exit_code=0 if item["report"] else 1)
    receipt={"entity":entity,"project":project,"project_url":f"https://wandb.ai/{entity}/{project}","runs":urls}
    (folder/"wandb-sync.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    return receipt


def maybe_sync(folder):
    if os.environ.get("REPAIR_WANDB")!="1":
        return
    try:
        result=sync(folder)
        print("W&B evidence: "+result["project_url"],flush=True)
    except Exception as exc:
        # Tracking availability must never determine whether policy weights pass.
        print("W&B sync incomplete ("+type(exc).__name__+"); saved experiment evidence is unchanged. Retry sync-wandb.",flush=True)
