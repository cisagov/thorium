#!/bin/bash

PCAP=$1
LOGS="zeek-logs-$(basename $1)"
mkdir -p /tmp/thorium/result-files/logs
mkdir -p /tmp/thorium/result-files/zeek-logs-$(basename $1)
zeek -C -r $PCAP Log::default_logdir=/tmp/thorium/result-files/logs
mv /tmp/thorium/result-files/logs/* /tmp/thorium/result-files/zeek-logs-$(basename $1)/
# save json logs and put them in logs directory (default name collides)
zeek -C -r $PCAP LogAscii::use_json=T Log::default_logdir=/tmp/thorium/result-files/logs
ls -1 /tmp/thorium/result-files/logs | xargs -I{} mv /tmp/thorium/result-files/logs/{} /tmp/thorium/result-files/$LOGS/{}.json
rm --dir /tmp/thorium/result-files/logs
cd  /tmp/thorium/result-files
tar -czvf /tmp/thorium/result-files/zeek-logs-$(basename $1).tar.gz ./$LOGS

dump_results() {
    RESULTS=$1

    echo "<pre>"
    # dns logs
    if [ -f "$RESULTS/dns.log" ]; then
        echo "<h5>Top DNS Queries (max=20)</h5>"
        echo -e "<b>Count \tHost</b>"
        cat $RESULTS/dns.log | zeek-cut query | sort | uniq -c | sort -nr | sed "s/ /\t/g" | head -20
        
        echo "<br/>" 
        echo "<h5>Bottom DNS Queries (max=20)</h5>"
        echo -e "<b>Count \tHost</b>"
        # reverse sort these
        cat $RESULTS/dns.log | zeek-cut query | sort -r | uniq -c | sort -n | sed "s/ /\t/g" | head -20
        echo "<br/>" 
    fi

    # connection logs
    if [ -f "$RESULTS/conn.log" ]; then
        echo "<h5>Connections Preview (max=100)</h5>"
        echo -e "<b>Count\tSource\t\tDest\t\tPort\tProto\tService</b>"
        cat $RESULTS/conn.log | zeek-cut id.orig_h id.resp_h id.resp_p proto service | sort | uniq -c | sort -nr | sed "s/ [ ]*/\t/g" | head -100
        echo "<br/>" 
    fi
    echo "</pre>"
    # next steps instructions
    echo "<p>To process the following <a href=https://docs.zeek.org/en/current/logs/index.html>Zeek logs</a>, you can use the zeek-cut utility. To install Zeek, follow these <a href=https://docs.zeek.org/en/current/install.html>instructions</a>. Alternatively, JSON formatted logs have been dumped as well, those can be processed with your preferred JSON parser.</p>"
}

dump_results /tmp/thorium/result-files/$LOGS | sed 's/^[ \t]*//' >> /tmp/thorium/results
rm -rf /tmp/thorium/result-files/$LOGS
