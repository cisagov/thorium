import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Alert, Button, Card, Col, Modal, Row, Tab, Tabs } from 'react-bootstrap';
import { FaTrash } from 'react-icons/fa';

// project imports
import { getStatusBadge, getStatusIcon, OverlayTipBottom, OverlayTipLeft, OverlayTipTop, Subtitle, Page } from '@components';
import { useAuth } from '@utilities';
import { deleteReaction, getPipeline, getReaction, getReactionLogs } from '@thorpi';

const ReactionStatus = () => {
  const { reactionID } = useParams();
  const { group } = useParams();
  const [reactionInfo, setReactionInfo] = useState({});
  const [reactionInfoError, setReactionInfoError] = useState('');
  const [reactionLogs, setReactionLogs] = useState([]);
  const [pipelineOrder, setPipelineOrder] = useState([]);
  const [statusMap, setStatusMap] = useState({});
  const [reactionFinished, setReactionFinished] = useState(false);
  const [width, setWindowWidth] = useState(0);
  const { checkCookie } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletionStatus, setDeletionStatus] = useState('');

  useEffect(() => {
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [deletionStatus]);

  const updateDimensions = () => {
    const width = window.innerWidth;
    setWindowWidth(width);
  };

  // This is a temporary hack to get the job ID/stage name mapping, that
  // will be replaced by an equivalent API route. This could become very
  // slow in the case of large reaction logs.
  const parseReactionLogs = (reactionLogs) => {
    const stages = {};
    const status = {};
    reactionLogs.map((entry, idx) => {
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
          // status[`${entry.update.stage}`] = 'Completed';
          status[`${stages[entry.update.job]}`] = 'Completed';
          break;
        case 'JobFailed':
          // status[`${entry.update.stage}`] = 'Failed';
          status[`${stages[entry.update.job]}`] = 'Failed';
          break;
        case 'JobRunning':
          // status[`${entry.update.stage}`] = 'Running';
          status[`${entry.update.worker.split('-')[0]}`] = 'Running';
          break;
      }
    });
    setStatusMap(status);
  };

  // Get reaction details and status
  const getReactionInfo = async () => {
    const reaction = await getReaction(group, reactionID, setReactionInfoError);
    if (reaction) {
      setReactionInfo(reaction);
      if (reaction && reaction.status && reaction.status == 'Completed') {
        setReactionFinished(true);
      }
      if (reaction && reaction.pipeline) {
        const pipeline = await getPipeline(group, reaction.pipeline, checkCookie);
        if (pipeline && pipeline.order) {
          setPipelineOrder(pipeline.order);
        }
      }
    }
  };

  const getReactionStatusLogs = async () => {
    const logs = [];
    let moreLogs = true;
    let cursor = 0;
    // need to get all reactions in chunks of 100 until there are no more left
    while (moreLogs) {
      const requestedLogs = await getReactionLogs(group, reactionID, checkCookie, cursor, 1000);
      if (requestedLogs) {
        // add returned reactions to local reactions array
        logs.push(...requestedLogs);
        // if cursor is undefined there are no more reactions for this group/tag
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

  const handleShowDeleteModal = () => {
    setDeletionStatus('');
    setShowDeleteModal(true);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  // handle removal of reaction using trash button
  const handleRemoveClick = async () => {
    const res = await deleteReaction(group, reactionID, setDeletionStatus);
    if (res) {
      setDeletionStatus('Success');
    }
    setShowDeleteModal(false);
  };

  // Get logs on first page load
  useEffect(() => {
    getReactionInfo();
    getReactionStatusLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Get reaction info and logs every 5 seconds after initial load
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (!reactionFinished) {
        getReactionInfo();
        getReactionStatusLogs();
      }
    }, 5000);
    return () => clearInterval(intervalId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reactionFinished]);

  const renderUpdate = (reaction) => {
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

  const ReactionLogs = ({ logs }) => {
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

  const renderPipelineChart = (order, id, group) => {
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
        <Alert variant="success" className="d-flex justify-content-center">
          Reaction deleted successfully! Return to sample &nbsp;
          {reactionInfo.samples &&
            reactionInfo.samples.map((sample) => (
              <Link key={sample} to={`/file/${sample}`}>
                {width <= 768 && sample.length > 15 ? sample.substring(0, 15) + '...' : sample}
              </Link>
            ))}
        </Alert>
      ) : reactionInfoError ? (
        <Alert variant="warning" className="d-flex justify-content-center">
          {'Error: ' + reactionInfoError}
        </Alert>
      ) : (
        reactionInfo.id && (
          <>
            {deletionStatus && (
              <Alert variant="danger" className="d-flex justify-content-center">
                {deletionStatus}
              </Alert>
            )}
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
                              <Button size="md" variant="" className="icon-btn" disabled={false} onClick={() => handleShowDeleteModal()}>
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
                                <Button className="danger-btn" onClick={handleRemoveClick}>
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
                          <Col>{getStatusBadge(reactionInfo.status)}</Col>
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
                        {Object(reactionInfo.generators).length > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Generators</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.generators}</Col>
                          </Row>
                        )}
                        {Object(reactionInfo.ephemeral).length > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Ephemeral</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.ephemeral}</Col>
                          </Row>
                        )}
                        {reactionInfo.parent_ephemeral > 0 && Object.keys(reactionInfo.parent_ephemeral).length && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Parent Ephemeral</Subtitle>
                            </Col>
                            <Col xs={10}>{JSON.stringify(reactionInfo.parent_ephemeral, null, 2)}</Col>
                          </Row>
                        )}
                        {Object(reactionInfo.repos).length > 0 && (
                          <Row>
                            <Col xs={2}>
                              <Subtitle>Repos</Subtitle>
                            </Col>
                            <Col xs={10}>{reactionInfo.repos}</Col>
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
                          <Col xs={9}>{getStatusBadge(reactionInfo.status)}</Col>
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
                        {Object(reactionInfo.generators).length > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Generators</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.generators}</Col>
                          </Row>
                        )}
                        {Object(reactionInfo.ephemeral).length > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Ephemeral</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.ephemeral}</Col>
                          </Row>
                        )}
                        {reactionInfo.parent_ephemeral > 0 && Object.keys(reactionInfo.parent_ephemeral).length && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Parent Ephemeral</Subtitle>
                            </Col>
                            <Col xs={9}>{JSON.stringify(reactionInfo.parent_ephemeral, null, 2)}</Col>
                          </Row>
                        )}
                        {Object(reactionInfo.repos).length > 0 && (
                          <Row className="flex-nowrap">
                            <Col className="reaction-name-width" xs={2}>
                              <Subtitle>Repos</Subtitle>
                            </Col>
                            <Col xs={9}>{reactionInfo.repos}</Col>
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
                    {renderPipelineChart(pipelineOrder, reactionInfo.id, reactionInfo.group)}
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
