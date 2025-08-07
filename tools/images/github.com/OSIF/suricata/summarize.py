import json
import argparse

parser = argparse.ArgumentParser("Summarize Suricata Logs")
parser.add_argument("--logs", help="Path to Suricata logs directory", default="/tmp/thorium/result-files")
parser.add_argument("--results", help="Path result file for alert summary", default="/tmp/thorium/results")
parser.add_argument("--result-files", help="Path to directory to dump result files", default="/tmp/thorium/result-files")
parser.add_argument("--tags", help="Path to file to dump Thorium tags", default="/tmp/thorium/tags")
args = parser.parse_args()

def parse_event_logs(path):
  alerts = []
  alert_summary = {}
  stats = {}

  with open(path) as file:
    for line in file:
      event = json.loads(line)
      if "event_type" in event:
        event_type = event["event_type"]
        if event_type == "alert":
            alerts.append(event)
            sig_id = event['alert']['signature_id']
            signature = event['alert']['signature']
            severity = event['alert']['severity']
            timestamp = event['timestamp']
            if sig_id in alert_summary:
              alert_summary[sig_id]['count'] += 1
              alert_summary[sig_id]['timestamps'].append(timestamp)
              #print(f"Saving duplicate alert: {event}") 
            else:
              alert_summary[sig_id] = {"count": 1, "signature": signature, "severity": severity, "timestamps": [timestamp]}
              #print(f"Saving new alert: {event}") 
        elif event_type == "stats":
            stats = event
            #print(f"Saving stats for pcap: {event}")
  return (alert_summary, stats)

def summarize_alerts(alert_summary):
  return sorted(alert_summary.items(), key=lambda item: (item[1]['severity'], -item[1]['count']))

def dump_tags(alert_symmary):
  alerts = []
  for k, v in alert_summary.items():
    alerts.append(f"{v['signature']} ({k})")
  return {"SuricataAlert": alerts}

def dump_table(sorted_alert_summary):
  table = "Signature ID,Signature,Severity,Count,Timestamps\n"

  for (sig_id, sig_info) in sorted_alert_summary:
    table += f"{sig_id},{sig_info['signature']},{sig_info['severity']},{sig_info['count']},{sig_info['timestamps']}\n" 
  return table

alert_summary, stats = parse_event_logs(f"{args.logs}/eve.json")
sorted_alert_summary = summarize_alerts(alert_summary)
alert_table = dump_table(sorted_alert_summary)
tags = dump_tags(alert_summary)

with open(args.tags, 'w') as f:
  f.write(json.dumps(tags, indent=4))
with open(f"{args.result_files}/alert_summary.json", 'w') as f:
  f.write(json.dumps(sorted_alert_summary, indent=4))
with open(args.results, 'w') as f:
  f.write("### Suricata Alerts\n")
  f.write(alert_table)
  # drop stats csv table into results
  f.write("\n### PCAP Stats\n")
