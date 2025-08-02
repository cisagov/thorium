import React, { useEffect, useState, Fragment } from 'react';
import { Alert, Col, Form, Row, Table } from 'react-bootstrap';

// project imports
import { Subtitle, Title, Page } from '@components';
import { getSystemStats } from '@thorpi';

const SelectableGroupStats = ({ stats }) => {
  const groups = stats ? Object.keys(stats) : [];
  const initialGroup = groups.length > 0 ? groups.sort()[0] : '';
  const initialPipeline =
    initialGroup && Object.keys(stats[initialGroup]['pipelines']).length > 0 ? Object.keys(stats[initialGroup]['pipelines']).sort()[0] : '';
  const [selectedGroup, setSelectedGroup] = useState(initialGroup);
  const [selectedPipeline, setSelectedPipeline] = useState(initialPipeline);
  const [pipelineStats, setPipelineStats] = useState({});

  useEffect(() => {
    if (selectedGroup) {
      const defaultPipeline =
        Object.keys(stats[selectedGroup]['pipelines']).length > 0 ? Object.keys(stats[selectedGroup]['pipelines']).sort()[0] : '';
      setSelectedPipeline(defaultPipeline);
      setPipelineStats(stats[selectedGroup]['pipelines'][defaultPipeline]);
    }
  }, [selectedGroup, stats, setPipelineStats, setSelectedPipeline]);

  if (!stats) {
    return null;
  } else {
    return (
      <>
        <Form.Group>
          <Form.Label>
            <Subtitle>Group</Subtitle>
          </Form.Label>
          <Form.Select value={selectedGroup} onChange={(e) => setSelectedGroup(e.target.value)}>
            {groups && groups.map((name) => <option key={`${name}_group_option`}>{name}</option>)}
          </Form.Select>
          <Form.Label>
            <Subtitle>Pipeline</Subtitle>
          </Form.Label>
          <Form.Select
            value={selectedPipeline}
            onChange={(e) => {
              // set the selected pipeline name
              setSelectedPipeline(e.target.value);
              // set the stats for the selected pipeline
              setPipelineStats(stats[selectedGroup]['pipelines'][e.target.value]);
            }}
          >
            {selectedPipeline &&
              Object.keys(stats[selectedGroup]['pipelines'])
                .sort()
                .map((name) => <option key={`${selectedGroup}_${name}_option`}>{name}</option>)}
          </Form.Select>
        </Form.Group>
        {pipelineStats && Object.keys(pipelineStats).length > 0 && (
          <Row>
            <Col>
              <pre className="text">{JSON.stringify(pipelineStats, null, '  ')}</pre>
            </Col>
          </Row>
        )}
      </>
    );
  }
};

const SystemStats = () => {
  const [getStatsError, setGetStatsError] = useState('');
  const [systemStats, setSystemStats] = useState({});
  const fetchStats = async () => {
    const stats = await getSystemStats(setGetStatsError);
    if (stats) {
      setSystemStats(stats);
    }
  };

  // trigger fetch stats on initial page load
  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <Page title="Stats · Thorium">
      {getStatsError != '' && (
        <Row>
          <Alert>{getStatsError}</Alert>
        </Row>
      )}
      <Row>
        <Col className="d-flex justify-content-center">
          <Title>System</Title>
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center">
          <Table striped bordered>
            <tbody>
              <tr>
                <td>
                  <Subtitle>Deadlines</Subtitle>
                </td>
                <td>
                  <center>
                    <Subtitle>{systemStats['deadlines']}</Subtitle>
                  </center>
                </td>
              </tr>
              <tr>
                <td>
                  <Subtitle>Running</Subtitle>
                </td>
                <td>
                  <center>
                    <Subtitle>{systemStats['running']}</Subtitle>
                  </center>
                </td>
              </tr>
              <tr>
                <td>
                  <Subtitle>Users</Subtitle>
                </td>
                <td>
                  <center>
                    <Subtitle>{systemStats['users']}</Subtitle>
                  </center>
                </td>
              </tr>
            </tbody>
          </Table>
        </Col>
      </Row>
      <br />
      <Row>
        <Col className="d-flex justify-content-center">
          <Title>Scaler</Title>
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center">
          <Table striped bordered>
            <tbody>
              {Object.keys(systemStats)
                .filter((field) => field != 'groups')
                .map((field) => {
                  if (['k8s', 'baremetal', 'external'].includes(field)) {
                    return (
                      <Fragment key={field}>
                        <tr>
                          <td rowSpan={2}>
                            <Subtitle>{field}</Subtitle>
                          </td>
                          <td>
                            <center>
                              <Subtitle>Deadlines</Subtitle>
                            </center>
                          </td>
                          <td>
                            <center>
                              <Subtitle>{systemStats[field].deadlines}</Subtitle>
                            </center>
                          </td>
                        </tr>
                        <tr>
                          <td>
                            <center>
                              <Subtitle>Running</Subtitle>
                            </center>
                          </td>
                          <td>
                            <center>
                              <Subtitle>{systemStats[field].running}</Subtitle>
                            </center>
                          </td>
                        </tr>
                      </Fragment>
                    );
                  } else {
                    return null;
                  }
                })}
            </tbody>
          </Table>
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center">
          <Title>Pipelines</Title>
        </Col>
      </Row>
      <Row>
        <Col>
          <SelectableGroupStats stats={systemStats.groups} />
        </Col>
      </Row>
    </Page>
  );
};

export default SystemStats;
