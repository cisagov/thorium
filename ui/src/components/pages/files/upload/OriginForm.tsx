import React from 'react';
import { Card, Tab, Tabs } from 'react-bootstrap';
import { useUpload } from './UploadContext';
import { OriginType } from './types';
import OriginDownloaded from './OriginDownloaded';
import OriginParentToolFlags from './OriginParentToolFlags';
import OriginCarved from './OriginCarved';
import OriginWire from './OriginWire';
import OriginIncident from './OriginIncident';
import OriginMemoryDump from './OriginMemoryDump';

const OriginForm: React.FC = () => {
  const { originState, origin } = useUpload();

  return (
    <Card className="panel">
      <Card.Body>
        <Tabs fill activeKey={originState.originType} onSelect={(k) => origin.setOriginType((k || 'Downloaded') as OriginType)}>
          <Tab eventKey="Downloaded" title="Downloaded">
            <OriginDownloaded />
          </Tab>
          <Tab eventKey="Transformed" title="Transformed">
            <OriginParentToolFlags variant="transformed" />
          </Tab>
          <Tab eventKey="Unpacked" title="Unpacked">
            <OriginParentToolFlags variant="unpacked" />
          </Tab>
          <Tab eventKey="Carved" title="Carved">
            <OriginCarved />
          </Tab>
          <Tab eventKey="Wire" title="Wire">
            <OriginWire />
          </Tab>
          <Tab eventKey="Incident" title="Incident">
            <OriginIncident />
          </Tab>
          <Tab eventKey="MemoryDump" title="Memory Dump">
            <OriginMemoryDump />
          </Tab>
        </Tabs>
      </Card.Body>
    </Card>
  );
};

export default OriginForm;
