# CEDIA Utilities

These optional tools synchronize a separately maintained V4 training tree and submit or inspect SLURM jobs. They do not contain project credentials or workstation-specific paths.

## Prerequisites

Install the optional dependency from the repository root:

```powershell
python -m pip install -r requirements-cedia.txt
```

Configure these variables in the current shell or an approved local secret manager. Do not put private values in tracked files.

| Variable | Required | Purpose |
|---|---|---|
| `TITAN_CEDIA_USERNAME` | Yes | Authorized CEDIA account. |
| `TITAN_CEDIA_LOCAL_V4` | For upload/sync | Local root of the separate `V4 CEDIA` project tree. |
| `TITAN_CEDIA_REMOTE_V4` | Yes | Remote project root allocated to the account. |
| `TITAN_CEDIA_HOST` | No | SSH host; defaults to `hpc.cedia.edu.ec`. |
| `TITAN_CEDIA_KEY_FILE` | No | Private-key path; defaults to `~/.ssh/cedia_rsa`. |
| `TITAN_CEDIA_KNOWN_HOSTS` | No | Additional verified known-hosts file. System SSH host keys are loaded as well. |

Unknown SSH host keys are rejected. Before first use, obtain the CEDIA host-key fingerprint from an independent CEDIA administrator channel and add the verified key to the system `known_hosts` file or the configured known-hosts file. Do not trust a key solely because `ssh-keyscan` returned it.

## Commands

From the repository root, use `python -m cedia.launch_training --no_launch` to synchronize and run the preflight audit without submitting training. The default launch command submits a preflight job and then a dependent training job; review its arguments and remote target before omitting `--no_launch`.

Use `python -m cedia.launch_training --status --skip_sync` to inspect the configured remote job, or `python -m cedia.launch_training --cancel_job JOB_ID --skip_sync` to cancel a job. `python -m cedia.sync_data` uploads the configured data delta. `python -m cedia.tail_log path/relative/to/project.log` reads a log under the configured remote project root. `python -m cedia.run_job --command 'squeue -u "$USER"'` runs an explicitly supplied remote shell command.

These commands were not run against CEDIA during the repository audit. Remote connectivity, host-key enrollment, account authorization, and SLURM behavior remain to be verified by the operator.
