"""Run with the Prime CLI's WSL Python; no credentials are printed or saved."""
import json
import sys
from pathlib import Path
from prime_cli.api.inference import InferenceClient, InferenceAPIError

payload = {"model": "qwen/qwen3-235b-a22b-2507",
           "messages": [{"role":"user","content":"Reply with OK only."}],
           "temperature":0,"max_tokens":8}
client = InferenceClient(timeout=30)
override = "--override" in sys.argv
if override:
    # Test an explicitly named provider from the gateway's own available list.
    payload["provider"] = {"only": ["deepinfra"], "allow_fallbacks": True}
report = {"endpoint":client.inference_url,"request_fields":list(payload),
          "provider_field_sent":override,"requested_provider": "deepinfra" if override else None,
          "local_provider_allowlist":False,
          "working_model_used_for_experiment":"qwen/qwen3.6-35b-a3b"}
try:
    reply = client.chat_completion(payload)
    report.update(status="working",reply=reply["choices"][0]["message"].get("content"))
except InferenceAPIError as exc:
    report.update(status="gateway_error",error=str(exc))
except Exception as exc:
    report.update(status="transport_error",error_type=type(exc).__name__)
name = "prime-routing-override-test.json" if override else "prime-routing-diagnostic.json"
path=Path(__file__).resolve().parents[1]/"artifacts"/name
path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
