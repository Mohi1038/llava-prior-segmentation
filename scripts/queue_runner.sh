#!/bin/bash
# Sequential SLURM queue for this project: submits queue/tasks.txt one task at a time,
# YIELDS TO RESEARCH JOBS (user priority, 2026-10-27): any of the user's jobs that is not a task in
# queue/tasks.txt counts as research. While one is pending or running, nothing new is submitted;
# if one is PENDING (waiting for a QOS slot), this project's queued/running job is cancelled and its
# task returned to pending without using an attempt (every task resumes from its cache).
# starts the next when one COMPLETES, resubmits a failed task (commands resume from their
# caches) up to MAX_ATTEMPTS, and stops the queue if a task keeps failing.
#
#   queue/tasks.txt   one task per line:  name|shell command (run via slurm/run.slurm)
#   queue/state.tsv   name  status(pending|running|done|failed)  jobid  attempts
#   queue/STATUS.txt  human-readable summary, rewritten every cycle
#   queue/runner.log  event log
#
# Start (idempotent; cron calls this every 10 min so the watch survives logouts):
#   scripts/queue_runner.sh start
set -u
P=/scratch/b23cs1106/projects/llava-prior-segmentation
Q=$P/queue
MAX_ATTEMPTS=3
POLL=60
mkdir -p "$Q"
cd "$P" || exit 1

log() { echo "$(date '+%F %T') $*" >> "$Q/runner.log"; }

if [ "${1:-}" = "start" ]; then
    if [ -f "$Q/runner.pid" ] && kill -0 "$(cat "$Q/runner.pid")" 2>/dev/null; then
        exit 0
    fi
    nohup setsid "$0" loop >> "$Q/runner.log" 2>&1 < /dev/null &
    echo $! > "$Q/runner.pid"
    log "runner started (pid $!)"
    exit 0
fi
[ "${1:-}" = "loop" ] || { echo "usage: $0 start"; exit 2; }

touch "$Q/state.tsv"
# Register new tasks as pending.
sync_tasks() {
    while IFS='|' read -r name cmd; do
        [ -z "$name" ] && continue
        case "$name" in \#*) continue ;; esac
        grep -q "^$name	" "$Q/state.tsv" || printf '%s\tpending\t-\t0\n' "$name" >> "$Q/state.tsv"
    done < "$Q/tasks.txt"
}
set_state() {  # name status jobid attempts
    awk -v n="$1" -v s="$2" -v j="$3" -v a="$4" 'BEGIN{FS=OFS="\t"} $1==n{$2=s;$3=j;$4=a} {print}' \
        "$Q/state.tsv" > "$Q/state.tmp" && mv "$Q/state.tmp" "$Q/state.tsv"
}
task_cmd() { grep "^$1|" "$Q/tasks.txt" | head -1 | cut -d'|' -f2-; }
write_status() {
    {
        echo "LLaVA-prior queue status — $(date '+%F %T')"
        echo
        printf '%-22s %-9s %-8s %s\n' TASK STATUS JOBID ATTEMPTS
        awk 'BEGIN{FS="\t"} {printf "%-22s %-9s %-8s %s\n",$1,$2,$3,$4}' "$Q/state.tsv"
        running=$(awk -F'\t' '$2=="running"{print $3}' "$Q/state.tsv")
        if [ -n "$running" ]; then
            name=$(awk -F'\t' '$2=="running"{print $1}' "$Q/state.tsv")
            echo
            echo "latest log line ($name):"
            tail -n 1 "$P/logs/$name-$running.out" 2>/dev/null | tr '\r' '\n' | tail -n 1
        fi
    } > "$Q/STATUS.txt"
}

task_names() { grep -v '^#' "$Q/tasks.txt" | cut -d'|' -f1; }
research_jobs() {  # "jobid name state" for the user's jobs that are not queue tasks
    squeue -u "$USER" -h -o "%i %j %T" 2>/dev/null | while read -r id name st; do
        task_names | grep -qx "$name" || echo "$id $name $st"
    done
}

log "loop running"
while true; do
    sync_tasks
    research=$(research_jobs)
    if [ -n "$research" ]; then
        if echo "$research" | grep -q " PENDING$"; then
            ours=$(awk -F'\t' '$2=="running"' "$Q/state.tsv" | head -1)
            if [ -n "$ours" ]; then
                name=$(echo "$ours" | cut -f1); jobid=$(echo "$ours" | cut -f3); attempts=$(echo "$ours" | cut -f4)
                scancel "$jobid"
                set_state "$name" pending "$jobid" "$((attempts - 1))"
                log "$name ($jobid) cancelled to free the GPU for research job(s): $(echo "$research" | tr '\n' ' ')"
            fi
        fi
        [ "$(cat "$Q/yield.flag" 2>/dev/null)" = "$research" ] || { log "waiting for research job(s): $(echo "$research" | tr '\n' ' ')"; echo "$research" > "$Q/yield.flag"; }
        write_status; sleep "$POLL"; continue
    fi
    rm -f "$Q/yield.flag"
    running=$(awk -F'\t' '$2=="running"' "$Q/state.tsv" | head -1)
    if [ -n "$running" ]; then
        name=$(echo "$running" | cut -f1); jobid=$(echo "$running" | cut -f3); attempts=$(echo "$running" | cut -f4)
        state=$(sacct -j "$jobid" -n -X -o State 2>/dev/null | head -1 | tr -d ' +')
        case "$state" in
            COMPLETED)
                set_state "$name" done "$jobid" "$attempts"; log "$name ($jobid) COMPLETED" ;;
            FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
                log "$name ($jobid) $state; last lines: $(tail -n 3 "$P/logs/$name-$jobid.out" 2>/dev/null | tr '\n\r' '  ')"
                if [ "$attempts" -lt "$MAX_ATTEMPTS" ]; then
                    set_state "$name" pending "$jobid" "$attempts"; log "$name will be resubmitted"
                else
                    set_state "$name" failed "$jobid" "$attempts"; log "$name FAILED $attempts times; queue stopped"
                fi ;;
            *) ;;  # PENDING / RUNNING / unknown yet
        esac
    else
        if awk -F'\t' '$2=="failed"' "$Q/state.tsv" | grep -q .; then
            write_status; sleep "$POLL"; continue   # wait for a human/Claude fix
        fi
        next=$(awk -F'\t' '$2=="pending"' "$Q/state.tsv" | head -1)
        if [ -n "$next" ]; then
            name=$(echo "$next" | cut -f1); attempts=$(echo "$next" | cut -f4)
            cmd=$(task_cmd "$name")
            out=$(sbatch -J "$name" slurm/run.slurm "$cmd" 2>&1)
            jobid=$(echo "$out" | grep -oE '[0-9]+$')
            if [ -n "$jobid" ]; then
                set_state "$name" running "$jobid" "$((attempts + 1))"; log "$name submitted as $jobid (attempt $((attempts + 1)))"
            else
                log "sbatch for $name refused: $out"   # e.g. QOS limit: retry next cycle
            fi
        fi
    fi
    write_status
    sleep "$POLL"
done
