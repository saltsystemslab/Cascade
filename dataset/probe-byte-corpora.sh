#!/usr/bin/env bash
# Probe every URL in byte-corpora.tsv for reachability and size. Downloads
# nothing: a HEAD, falling back to a one-byte ranged GET.
#
#   bash dataset/probe-byte-corpora.sh [path/to/byte-corpora.tsv]
#
# The fallback matters. Several of these hosts are old enough not to answer
# HEAD at all, while every HTTP/1.1 server answers a Range request -- and the
# Content-Range reply carries the true length, which HEAD often omits.
#
# Run this and correct the TSV from its output BEFORE writing any fetch logic.
# Expect at least one Pizza&Chili 404 on the first pass; that is the normal
# state of that corpus and the whole reason the URL list is data.
set -uo pipefail

TSV="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/byte-corpora.tsv}"
[ -f "$TSV" ] || { echo "no such file: $TSV" >&2; exit 2; }

CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
MAX_TIME="${MAX_TIME:-30}"

printf '%-18s %-6s %16s  %s\n' LABEL CODE BYTES URL
printf '%-18s %-6s %16s  %s\n' ------ ---- ----- ---

reachable=0
dead=0

while IFS=$'\t' read -r label urls archive member post expect dl_sha out_sha; do
    case "${label:-}" in ''|\#*) continue ;; esac

    got_one=0
    IFS='|' read -ra alts <<< "$urls"
    for u in "${alts[@]}"; do
        code=$(curl -sIL --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
                    -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || echo 000)

        # A 1-byte ranged GET: answered by every HTTP/1.1 server, and its
        # Content-Range gives the true length even when HEAD is refused.
        len=$(curl -sL -r 0-0 -D - -o /dev/null --connect-timeout "$CONNECT_TIMEOUT" \
                   --max-time "$MAX_TIME" "$u" 2>/dev/null \
              | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range:/ {print $2; exit}')

        if [ "$code" != 200 ] && [ -n "$len" ]; then
            code="$code~200"     # HEAD refused but the body is fetchable
        fi

        if [ -n "$len" ] || [ "$code" = 200 ]; then
            printf '%-18s %-6s %16s  %s\n' "$label" "$code" "${len:-?}" "$u"
            got_one=1
            reachable=$((reachable + 1))
            break
        fi
        printf '%-18s %-6s %16s  %s\n' "$label" "$code" "-" "$u"
    done

    if [ "$got_one" -eq 0 ]; then
        dead=$((dead + 1))
        printf '%-18s %-6s %16s  %s\n' "$label" "DEAD" "-" "(no alternate answered)"
    fi
done < "$TSV"

echo
echo "$reachable reachable, $dead with no working alternate"
echo
echo "BYTES is the COMPRESSED download size (Content-Range), not the artifact."
echo "expect_bytes in the TSV is the UNCOMPRESSED size and must be filled in"
echo "after a real fetch -- except enwik8/9, whose sizes are definitional."
[ "$dead" -eq 0 ]
