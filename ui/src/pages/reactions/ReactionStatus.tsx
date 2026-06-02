import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Button, Card, Col, Modal, Row, Tab, Tabs } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

// project imports
import { getStatusIcon, getStatusBadge } from '@components/pages/files/reactions';
import Page from '@components/pages/Page';
import Subtitle from '@components/shared/titles/Subtitle';
import { OverlayTipBottom, OverlayTipLeft, OverlayTipTop } from '@components/shared/overlay/tips';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { useAuth } from '@utilities/auth';
import { deleteReaction, getReaction, getReactionLogs } from '@thorpi/reactions';
import { getPipeline } from '@thorpi/pipelines';
import { ReactionStatus as ReactionStatusEnum, type Reaction, type ReactionLogEntry } from '@models/reactions';

interface ReactionLogsProps {
  logs: ReactionLogEntry[];
}

const ReactionStatus = () => {
  const { reactionID } = useParams();
  const { group } = useParams();
  const [reactionInfo, setReactionInfo] = useState<Partial<Reaction>>({});
  const [reactionInfoError, setReactionInfoError] = useState('');
  const [reactionLogs, setReactionLogs] = useState<ReactionLogEntry[]>([]);
  const [pipelineOrder, setPipelineOrder] = useState<string[][]>([]);
  const [statusMap, setStatusMap] = useState<Record<string, string>>({});
  const [reactionFinished, setReactionFinished] = useState(false);
  const [width, setWindowWidth] = useState(0);
  const { checkCookie } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletionStatus, setDeletionStatus] = useState('');

  // Update stored window width
  const updateDimensions = () => {
    const width = window.innerWidth;
    setWindowWidth(width);
  };

  useEffect(() => {
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [deletionStatus]);

  // This is a temporary hack to get the job ID/stage name mapping, that
  // will be replaced by an equivalent API route. This could become very
  // slow in the case of large reaction logs.
  const parseReactionLogs = (reactionLogs: ReactionLogEntry[]) => {
    const stages: Record<string, string> = {};
    const status: Record<string, string> = {};
    reactionLogs.map((entry) => {
      // set the status map based on the entry.action value
      switch (entry.action) {
        case 'JobCreated':
          if (stages[entry.update.id] == undefined) {
            stages[`${entry.update.id}`] = entry.update.stage;
            stages[`${entry.update.stage}`] = entry.update.id;
          }
          status[`${entry.update.stage}`] = 'Created';
          break;
        case 'JobCompleted':
          status[`${stages[entry.update.job]}`] = 'Completed';
          break;
        case 'JobFailed':
          status[`${stages[entry.update.job]}`] = 'Failed';
          break;
        case 'JobRunning':
          status[`${entry.update.worker.split('-')[0]}`] = 'Running';
          break;
      }
    });
    setStatusMap(status);
  };

  // Get reaction details and status
  const getReactionInfo = async () => {
    const reaction = await getReaction(group as string, reactionID as string, setReactionInfoError);
    if (reaction) {
      setReactionInfo(reaction);
      if (reaction && reaction.status && reaction.status === ReactionStatusEnum.Completed) {
        setReactionFinished(true);
      }
      if (reaction && reaction.pipeline) {
        const pipeline = await getPipeline(group as string, reaction.pipeline, () => {
          void checkCookie();
        });
        if (pipeline && pipeline.order) {
          setPipelineOrder(pipeline.order);
        }
      }
    }
  };

  // Fetch all reaction logs in paginated chunks
  const getReactionStatusLogs = async () => {
    const logs: ReactionLogEntry[] = [];
    let moreLogs = true;
    let cursor = 0;
    // need to get all reactions in chunks of 100 until there are no more left
    while (moreLogs) {
      const requestedLogs = await getReactionLogs(
        group as string,
        reactionID as string,
        () => {
          void checkCookie();
        },
        cursor as unknown as null,
        1000,
      );
      if (requestedLogs) {
        logs.push(...(requestedLogs as unknown as ReactionLogEntry[]));
        if (requestedLogs.length == 0) {
          moreLogs = false;
        } else {
          cursor += requestedLogs.length;
        }
      } else {
        moreLogs = false;
      }
    }
    setReactionLogs(logs);
    parseReactionLogs(logs);
  };

  // Show the delete confirmation modal
  const handleShowDeleteModal = () => {
    setDeletionStatus('');
    setShowDeleteModal(true);
  };

  // Close the delete confirmation modal
  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  // Handle removal of reaction using trash button
  const handleRemoveClick = async () => {
    const res = await deleteReaction(group as string, reactionID as string, setDeletionStatus);
    if (res) {
      setDeletionStatus('Success');
    }
    setShowDeleteModal(false);
  };

  // Get logs on first page load
  useEffect(() => {
    void getReactionInfo();
    void getReactionStatusLogs();
  }, []);

  // Get reaction info and logs every 5 seconds after initial load
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (!reactionFinished) {
        void getReactionInfo();
        void getReactionStatusLogs();
      }
    }, 5000);
    return () => clearInterval(intervalId);
  }, [reactionFinished]);

  // Render the key/value pairs for a single log entry update
  const renderUpdate = (reaction: ReactionLogEntry) => {
    const exclude = ['current_stage', 'group', 'pipeline', 'reaction', 'id', 'status'];
    const sorted = Object.keys(reaction.update)
      .filter((key) => !exclude.includes(key))
      .sort();
    return (
      <>
        {sorted.map((key) => (
          <Row key={key} className="flex-nowrap">
            <Col className="key-log" xs={2}>
              {key}
            </Col>
            <Col className="value-log">{reaction.update[key]}</Col>
          </Row>
        ))}
      </>
    );
  };

  // Display the reaction log entries in a scrollable card
  const ReactionLogs = ({ logs }: ReactionLogsProps) => {
    return (
      <Card className="log-box scroll-log panel">
        <Card.Header>
          <Row className="mt-1 mb-1 flex-nowrap">
            <Col className="timestamp-log" xs={3}>
              TIME STAMP
            </Col>
            <Col className="action-log" xs={2}>
              ACTION
            </Col>
            <Col className="flex-nowrap">
              <Row className="flex-nowrap">
                <Col className="key-log" xs={2}>
                  KEY
                </Col>
                <Col className="value-log">VALUE</Col>
              </Row>
            </Col>
          </Row>
        </Card.Header>
        <Card.Body>
          {logs.map((reaction, idx) => (
            <Row key={`${reaction.id}_${idx}`} className="flex-nowrap">
              <Col className="timestamp-log" xs={3}>
                {width <= 1400 ? reaction.timestamp.split('.')[0] : reaction.timestamp}
              </Col>
              <Col className="action-log" xs={2}>
                {reaction.action}
              </Col>
              <Col>{renderUpdate(reaction)}</Col>
            </Row>
          ))}
        </Card.Body>
      </Card>
    );
  };

  // Render the pipeline stage chart with status icons and links to stage logs
  const renderPipelineChart = (order: string[][], id: string, group: string) => {
    return (
      <Row className="ms-2 flex-nowrap pipeline-chart body-panel">
        {order.map &&
          order.map((stage, idx) => (
            <Col xs={2} key={idx} className="pipeline-col">
              <Row>
                {stage.map &&
                  stage.map((image, idx) => (
                    <Row key={`${image}_${idx}`}>
                      {statusMap[image] == undefined ? (
                        <Card className="m-1 panel reaction-card panel">
                          <Row>
                            <Col>{image}</Col>
                            <Col xs={2}>{getStatusIcon(statusMap[image])}</Col>
                          </Row>
                        </Card>
                      ) : (
                        <Link to={`/reaction/logs/${group}/${id}/${image}`} className="p-0 no-decoration">
                          <OverlayTipBottom tip={`Click to view the logs for ${image}`}>
                            <Card className="m-1 panel reaction-card">
                              <Row>
                                <Col>{image}</Col>
                                <Col xs={2}>{getStatusIcon(statusMap[image])}</Col>
                              </Row>
                            </Card>
                          </OverlayTipBottom>
                        </Link>
                      )}
                    </Row>
                  ))}
              </Row>
            </Col>
          ))}
      </Row>
    );
  };

  return (
    <Page title="Reaction Status" className="full-min-width">
      {deletionStatus == 'Success' ? (
        <AlertBanner severity={Severity.Success}>
          Reaction deleted successfully! Return to sample &nbsp;
          {reactionInfo.samples &&
            reactionInfo.samples.map((sample) => (
              <Link key={sample} to={`/file/${sample}`}>
                {width <= 768 && sample.length > 15 ? sample.substring(0, 15) + '...' : sample}
              </Link>
            ))}
        </AlertBanner>
      ) : reactionInfoError ? (
        <AlertBanner severity={Severity.Warning}>{'Error: ' + reactionInfoError}</AlertBanner>
      ) : (
        reactionInfo.id && (
          <>
            {deletionStatus && <AlertBanner>{deletionStatus}</AlertBanner>}
            <Row>
              <Col>
                <Card className="panel">
                  <Card.Body>
                    <Row>
                      <Col className="full-reactions-row" xs={5}>
                        <Row>
                          <Col xs={3}>
                            <Subtitle>Reaction ID</Subtitle>
                          </Col>
                          <Col>{reactionInfo.id}</Col>
                        </Row>
                        <br />
                        <Row>
                          <Col>
                            <OverlayTipTop
                              tip={`Delete this reaction. Only system admins, group
                              owners/managers, and the submitter can delete a reaction.`}
                            >
                              <Button size="sm" variant="" className="icon-btn" disabled={false} onClick={() => handleShowDeleteModal()}>
                                <FaTrash />
                              </Button>
                            </OverlayTipTop>
                            <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
                              <Modal.Header closeButton>
                                <Modal.Title>Confirm deletion?</Modal.Title>
                              </Modal.Header>
                              <Modal.Body>
                                <p>Do you really want to delete the reaction:</p>
                                <center>
                                  <p>
                                    <b>
                                      {reactionInfo.pipeline}&nbsp;:&nbsp;
                                      {reactionInfo.group}
                                    </b>
                                  </p>
                                </center>
                              </Modal.Body>
                              <Modal.Footer className="d-flex justify-content-center">
                                <Button
                                  className="danger-btn"
                                  onClick={() => {
                                    void handleRemoveClick();
                                  }}
                                >
                                  Confirm
                                </Button>
                                <Button className="primary-btn" onClick={handleCloseDeleteModal}>
                                  Cancel
                                </Button>
                              </Modal.Footer>
                            </Modal>
                          </Col>
                        </Row>
                        <br />
                        <Row>
                          <Col xs={3}>
                            <Subtitle>Status</Subtitle>
                          </Col>
                          <Col>{getStatusBadge(reactionInfo.status as string)}</Col>
                        </Row>
                      </Col>
                      <Col className="full-reactions-row" xs={7}>
                        <Row>
                          <Col xs={2}>
                            <Subtitle>Pipeline</Subtitle>
                          </Col>
                          <Col xs={10}>{reactionInfo.pipeline}</Col>
                        </Row>
                        <Row>
                          <Col xs={2}>
                            <Subtitle>Creator</Subtitle>
                          </Col>
                          <Col xs={10}>{reactionInfo.creator}</Col>
                        </Row>
                        <Row>
                          <Col xs={2}>
                            <Subtitle>Group</Subtitle>
                          </Col>
                          <Col xs={10}>{reactionInfo.group}</Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col xs={2}>
                            <Subtitle>Samples</Subtitle>
                          </Col>
                          <Col xs={10}>
                            {reactionInfo.samples &&
                              reactionInfo.samples.map((sample) => (
                                <OverlayTipLeft key={sample} tip={`Click to view details about this sample`}>
                                  <Row>
                                    <Link to={`/file/${sample}`}>
                                      {width <= 768 && sample.length > 15 ? sample.substring(0, 15) + '...' : sample}
                                    </Link>
                                  </Row>
                                </OverlayTipLeft>
                              ))}
                          </Col>
                        </Row>
                        <Row>
                          <Col xs={2}>
                            <Subtitle>SLA</Subtitle>
                          </Col>
                          <Col xs={10}>{reactionInfo.sla}</Col>
                        </Row>
                        <Row>
                          <Col xs={2}>
                            <Subtitle>Args</Subtitle>
                          </Col>
                          <Col xs={10}>{JSON.stringify(reactionInfo.args, null, 2)}</Col>
                        </Row>
                        {reactionInfo.parent && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Parent</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.parent}</Col>
                          </Row>
                        )}
                        {(reactionInfo.generators?.length ?? 0) > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Generators</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.generators?.join(', ')}</Col>
                          </Row>
                        )}
                        {(reactionInfo.ephemeral?.length ?? 0) > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Ephemeral</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.ephemeral?.join(', ')}</Col>
                          </Row>
                        )}
                        {reactionInfo.parent_ephemeral && Object.keys(reactionInfo.parent_ephemeral).length > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Parent Ephemeral</Subtitle>
                            </Col>
                            <Col xs={10}>{JSON.stringify(reactionInfo.parent_ephemeral, null, 2)}</Col>
                          </Row>
                        )}
                        {(reactionInfo.repos?.length ?? 0) > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Repos</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.repos?.map((r) => r.url).join(', ')}</Col>
                          </Row>
                        )}
                      </Col>
                      <Col className="compact-reactions-row" xs={7}>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Reaction ID</Subtitle>
                          </Col>
                          <Col xs={8}>{reactionInfo.id}</Col>
                        </Row>
                        <br />
                        <br />
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Status</Subtitle>
                          </Col>
                          <Col xs={9}>{getStatusBadge(reactionInfo.status as string)}</Col>
                        </Row>
                      </Col>
                      <Col className="compact-reactions-row" xs={7}>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Pipeline</Subtitle>
                          </Col>
                          <Col xs={9}>{reactionInfo.pipeline}</Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Creator</Subtitle>
                          </Col>
                          <Col xs={9}>{reactionInfo.creator}</Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Group</Subtitle>
                          </Col>
                          <Col xs={9}>{reactionInfo.group}</Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Samples</Subtitle>
                          </Col>
                          <Col xs={9}>
                            {reactionInfo.samples &&
                              reactionInfo.samples.map((sample) => (
                                <OverlayTipLeft key={sample} tip={`Click to view details about this sample`}>
                                  <Row>
                                    <Link to={`/file/${sample}`}>
                                      {width <= 768 && sample.length > 15 ? sample.substring(0, 15) + '...' : sample}
                                    </Link>
                                  </Row>
                                </OverlayTipLeft>
                              ))}
                          </Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>SLA</Subtitle>
                          </Col>
                          <Col xs={9}>{width <= 768 ? String(reactionInfo.sla).split(':')[0] : reactionInfo.sla}</Col>
                        </Row>
                        <Row className="flex-nowrap">
                          <Col className="reaction-name-width" xs={2}>
                            <Subtitle>Args</Subtitle>
                          </Col>
                          <Col xs={9}>{JSON.stringify(reactionInfo.args, null, 2)}</Col>
                        </Row>
                        {reactionInfo.parent && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Parent</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.parent}</Col>
                          </Row>
                        )}
                        {(reactionInfo.generators?.length ?? 0) > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Generators</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.generators?.join(', ')}</Col>
                          </Row>
                        )}
                        {(reactionInfo.ephemeral?.length ?? 0) > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Ephemeral</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.ephemeral?.join(', ')}</Col>
                          </Row>
                        )}
                        {reactionInfo.parent_ephemeral && Object.keys(reactionInfo.parent_ephemeral).length > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Parent Ephemeral</Subtitle>
                            </Col>
                            <Col xs={9}>{JSON.stringify(reactionInfo.parent_ephemeral, null, 2)}</Col>
                          </Row>
                        )}
                        {(reactionInfo.repos?.length ?? 0) > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Repos</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.repos?.map((r) => r.url).join(', ')}</Col>
                          </Row>
                        )}
                      </Col>
                    </Row>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
            <Row>
              <Col>
                <Tabs defaultActiveKey="pipeline" id="uncontrolled-tab-example" className="mb-3 mt-3">
                  <Tab eventKey="pipeline" title="Pipeline">
                    {renderPipelineChart(pipelineOrder, reactionInfo.id, reactionInfo.group as string)}
                  </Tab>
                  <Tab eventKey="logs" title="Logs">
                    <ReactionLogs logs={reactionLogs} />
                  </Tab>
                  <Tab eventKey="tags" title="Tags">
                    <Row>
                      <Col>
                        {reactionInfo.tags &&
                          reactionInfo.tags.map((tag, idx) => (
                            <Row key={`tag_${tag}_${idx}`}>
                              <Col>{width <= 768 && tag.length > 35 ? tag.substring(0, 35) + '...' : tag}</Col>
                            </Row>
                          ))}
                      </Col>
                    </Row>
                  </Tab>
                </Tabs>
              </Col>
            </Row>
          </>
        )
      )}
    </Page>
  );
};

export default ReactionStatus;
