import React from 'react';
import { Card, Row, Tab, Tabs } from 'react-bootstrap';
import { isIP } from 'is-ip';
import { useUpload } from './UploadContext';
import { CarvedSubType } from './types';
import OriginField from './OriginField';

const isValidProtocol = (proto: string) => ['TCP', 'Tcp', 'tcp', 'UDP', 'Udp', 'udp'].includes(proto);

const OriginCarved: React.FC = () => {
  const { originState, origin } = useUpload();
  const { parentFile, tool, carvedType, pcap } = originState.carved;

  return (
    <>
      <OriginField
        label="Parent"
        value={parentFile}
        onChange={(v) => origin.setCarvedField('parentFile', v)}
        placeholder="SHA256"
        isInvalid={!parentFile && !!tool}
        feedback="Please enter a SHA256 value for the Parent."
      />
      <OriginField label="Tool" value={tool} onChange={(v) => origin.setCarvedField('tool', v)} />
      <Row>
        <Card className="panel" style={{ boxShadow: 'none', border: 'none' }}>
          <Card.Body>
            <Tabs fill activeKey={carvedType} onSelect={(k) => origin.setCarvedType((k || 'Pcap') as CarvedSubType)}>
              <Tab eventKey="Pcap" title="PCAP">
                <OriginField
                  label="Source IP"
                  value={pcap.sourceIp}
                  onChange={(v) => origin.setCarvedPcapField('sourceIp', v)}
                  isInvalid={!!pcap.sourceIp && !isIP(pcap.sourceIp)}
                  feedback="Please enter a valid IPv4/IPv6 address."
                />
                <OriginField
                  label="Destination IP"
                  value={pcap.destinationIp}
                  onChange={(v) => origin.setCarvedPcapField('destinationIp', v)}
                  isInvalid={!!pcap.destinationIp && !isIP(pcap.destinationIp)}
                  feedback="Please enter a valid IPv4/IPv6 address."
                />
                <OriginField
                  label="Source Port"
                  value={pcap.sourcePort}
                  onChange={(v) => origin.setCarvedPcapField('sourcePort', v)}
                  type="number"
                  isInvalid={!!pcap.sourcePort && (Number(pcap.sourcePort) < 1 || Number(pcap.sourcePort) > 65535)}
                  feedback="Please enter a valid port (between 1 and 65535, inclusive)."
                />
                <OriginField
                  label="Destination Port"
                  value={pcap.destinationPort}
                  onChange={(v) => origin.setCarvedPcapField('destinationPort', v)}
                  type="number"
                  isInvalid={!!pcap.destinationPort && (Number(pcap.destinationPort) < 1 || Number(pcap.destinationPort) > 65535)}
                  feedback="Please enter a valid port (between 1 and 65535, inclusive)."
                />
                <OriginField
                  label="Protocol"
                  value={pcap.protocol}
                  onChange={(v) => origin.setCarvedPcapField('protocol', v)}
                  placeholder="TCP/UDP (optional)"
                  isInvalid={!!pcap.protocol && !isValidProtocol(pcap.protocol)}
                  feedback={'Please enter a valid protocol ("TCP/Tcp/tcp" or "UDP/Udp/udp").'}
                />
                <OriginField label="URL" value={pcap.url} onChange={(v) => origin.setCarvedPcapField('url', v)} />
              </Tab>
              <Tab eventKey="Unknown" title="Unknown"></Tab>
            </Tabs>
          </Card.Body>
        </Card>
      </Row>
    </>
  );
};

export default OriginCarved;
