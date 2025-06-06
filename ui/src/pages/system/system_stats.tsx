import { useEffect, useState, Fragment } from 'react';
import { Alert, Col, Form, Row, Table } from 'react-bootstrap';

// project imports
import { Subtitle, Title, Page } from '@components';
import { getSystemStats } from '@thorpi';
import { GroupsStats, PipelineStats, Stats } from 'models';

interface GroupStatsProps {
  stats: GroupsStats;
}

const SelectableGroupStats: React.FC<GroupStatsProps> = ({ stats }) => {
  const groups = stats ? Object.keys(stats) : [];
  const [searchQuery, setSearchQuery] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'group', direction: 'ascending' });

  const getSortedData = (data: any) => {
    const sortedData = [...data];
    sortedData.sort((a, b) => {
      if (a[sortConfig.key] < b[sortConfig.key]) {
        return sortConfig.direction === 'ascending' ? -1 : 1;
      }
      if (a[sortConfig.key] > b[sortConfig.key]) {
        return sortConfig.direction === 'ascending' ? 1 : -1;
      }
      return 0;
    });
    return sortedData;
  };

  const handleSort = (key: string) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const filteredData: any = [];
  groups.forEach((group) => {
    const pipelines = stats[group]['pipelines'];
    Object.keys(pipelines).forEach((pipeline) => {
      const stages = pipelines[pipeline]['stages'];
      Object.keys(stages).forEach((stage) => {
        const stageData = stages[stage];
        Object.keys(stageData).forEach((user) => {
          const userStats = stageData[user];
          filteredData.push({
            group,
            pipeline,
            stage,
            user,
            created: userStats.created,
            running: userStats.running,
            completed: userStats.completed,
            failed: userStats.failed,
            sleeping: userStats.sleeping,
            total: userStats.total,
          });
        });
      });
    });
  });

  const sortedData = getSortedData(
    filteredData.filter(
      (row: any) =>
        row.group.toLowerCase().includes(searchQuery.toLowerCase()) ||
        row.pipeline.toLowerCase().includes(searchQuery.toLowerCase()) ||
        row.stage.toLowerCase().includes(searchQuery.toLowerCase()) ||
        row.user.toLowerCase().includes(searchQuery.toLowerCase()),
    ),
  );

  if (!stats) {
    return null;
  } else {
    return (
      <>
        <Form.Group>
          <Form.Control type="text" placeholder="Search..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} />
        </Form.Group>
        <Row>
          <Col>
            <Table striped bordered className="mt-2">
              <thead>
                <tr>
                  <th onClick={() => handleSort('group')}>Group</th>
                  <th onClick={() => handleSort('pipeline')}>Stage Name</th>
                  <th onClick={() => handleSort('stage')}>Stage</th>
                  <th onClick={() => handleSort('user')}>User</th>
                  <th onClick={() => handleSort('created')}>Created</th>
                  <th onClick={() => handleSort('running')}>Running</th>
                  <th onClick={() => handleSort('completed')}>Completed</th>
                  <th onClick={() => handleSort('failed')}>Failed</th>
                  <th onClick={() => handleSort('sleeping')}>Sleeping</th>
                  <th onClick={() => handleSort('total')}>Total</th>
                </tr>
              </thead>
              <tbody>
                {sortedData.map((row, index) => (
                  <tr key={index}>
                    <td>{row.group}</td>
                    <td>{row.pipeline}</td>
                    <td>{row.stage}</td>
                    <td>{row.user}</td>
                    <td>{row.created}</td>
                    <td>{row.running}</td>
                    <td>{row.completed}</td>
                    <td>{row.failed}</td>
                    <td>{row.sleeping}</td>
                    <td>{row.total}</td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Col>
        </Row>
      </>
    );
  }
};

interface GlobalStatsProps {
  stats: Stats;
}

const GlobalStats: React.FC<GlobalStatsProps> = ({ stats }) => {
  return (
    <Table striped bordered>
      <tbody>
        <tr>
          <td>
            <Subtitle>Deadlines</Subtitle>
          </td>
          <td>
            <center>
              <Subtitle>{stats.deadlines}</Subtitle>
            </center>
          </td>
        </tr>
        <tr>
          <td>
            <Subtitle>Running</Subtitle>
          </td>
          <td>
            <center>
              <Subtitle>{stats.running}</Subtitle>
            </center>
          </td>
        </tr>
        <tr>
          <td>
            <Subtitle>Users</Subtitle>
          </td>
          <td>
            <center>
              <Subtitle>{stats.users}</Subtitle>
            </center>
          </td>
        </tr>
      </tbody>
    </Table>
  );
};

const ScalerStats: React.FC<GlobalStatsProps> = ({ stats }) => {
  const scalers: Array<keyof Pick<Stats, 'k8s' | 'baremetal' | 'external'>> = ['k8s', 'baremetal', 'external'];

  return (
    <Table striped bordered>
      <tbody>
        {scalers.map((scaler) => (
          <Fragment key={scaler}>
            <tr>
              <td rowSpan={2}>
                <Subtitle>{scaler}</Subtitle>
              </td>
              <td>
                <center>
                  <Subtitle>Deadlines</Subtitle>
                </center>
              </td>
              <td>
                <center>
                  <Subtitle>{stats[scaler].deadlines}</Subtitle>
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
                  <Subtitle>{stats[scaler].running}</Subtitle>
                </center>
              </td>
            </tr>
          </Fragment>
        ))}
      </tbody>
    </Table>
  );
};

const SystemStats = () => {
  const [getStatsError, setGetStatsError] = useState('');
  const [systemStats, setSystemStats] = useState<Stats | null>(null);
  const fetchStats = async () => {
    const stats = await getSystemStats(setGetStatsError);
    if (stats) {
      setSystemStats(stats);
    }
  };

  // trigger fetch stats on initial page load
  useEffect(() => {
    fetchStats();

    // set interval to rerun every 5 seconds
    const intervalId = setInterval(() => {
      fetchStats();
    }, 10000);

    // cleanup interval on component unmount
    return () => {
      clearInterval(intervalId);
    };
  }, []);

  return (
    <Page title="Stats · Thorium">
      {getStatsError != '' && (
        <Row>
          <Alert>{getStatsError}</Alert>
        </Row>
      )}
      {systemStats != null && (
        <>
          <Row>
            <Col className="d-flex justify-content-center">
              <Title>System</Title>
            </Col>
          </Row>
          <Row>
            <Col className="d-flex justify-content-center">
              <GlobalStats stats={systemStats} />
            </Col>
          </Row>
          <Row>
            <Col className="d-flex justify-content-center">
              <Title>Scaler</Title>
            </Col>
          </Row>
          <Row>
            <Col className="d-flex justify-content-center">
              <ScalerStats stats={systemStats} />
            </Col>
          </Row>
          <Row>
            <Col className="d-flex justify-content-center">
              <Title>Pipeline</Title>
            </Col>
          </Row>
          <Row>
            <Col>
              <SelectableGroupStats stats={systemStats.groups} />
            </Col>
          </Row>
        </>
      )}
    </Page>
  );
};

export default SystemStats;
