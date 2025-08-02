import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Badge, Button, ButtonToolbar, Card, Col, FormCheck, Modal, Row } from 'react-bootstrap';
import { FaCircle, FaCheckCircle, FaDotCircle, FaTimesCircle, FaSpinner, FaTrash } from 'react-icons/fa';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import { LoadingSpinner, OverlayTipTop, OverlayTipLeft, Subtitle, Title } from '@components';
import { useAuth } from '@utilities';
import { createReaction, deleteReaction, listPipelines, listReactions } from '@thorpi';

// not sure of a way to better avoid this (file) global
// Without it there isn't a good way to pause the auto refresh
// functionality.
let deleteInProgress = false;

// Get the colored badge based on the status of a given reaction/job
const getStatusBadge = (status) => {
  switch (status) {
    case 'Completed':
      return <Badge bg="success">Completed</Badge>;
    case 'Failed':
    case 'Errored':
      return <Badge bg="danger">Failed</Badge>;
    case 'Created':
      return <Badge bg="secondary">Created</Badge>;
    case 'Running':
      return <Badge bg="primary">Running</Badge>;
    default:
      return <Badge bg="secondary">{status}</Badge>;
  }
};

// Get the colored badge based on the status of a given reaction/job
const getStatusIcon = (status) => {
  switch (status) {
    case 'Completed':
      return <FaCheckCircle size={18} color="green" />;
    case 'Failed':
      return <FaTimesCircle size={18} color="red" />;
    case 'Created':
      return <FaDotCircle size={18} color="lightBlue" />;
    case 'Running':
      return <FaSpinner size={18} color="blue" />;
    default:
      return <FaCircle size={18} color="grey" />;
  }
};

// internal function to build a list of reactions from a pipelines details and a
// list of selected pipelines
const buildReactionsList = (selectedPipelines, tags) => {
  // build selected Jobs list
  const reactionList = [];
  Object.keys(selectedPipelines).map((group) => {
    Object.keys(selectedPipelines[group]).map((pipeline) => {
      if (selectedPipelines[group][pipeline]) {
        const body = {
          pipeline: pipeline,
          group: group,
          args: {},
          sla: 30,
        };
        if (tags != []) {
          body['tags'] = tags;
        }
        reactionList.push(body);
      }
    });
  });
  return reactionList;
};

// submit reactions for a sha256 for a partially build reaction list containing
// reaction info for the selected pipelines
const submitReactions = async (sha256, reactionList) => {
  const reactionRunResults = [];
  for (const reaction of reactionList) {
    reaction.samples = [sha256];

    // handle adding the error to the results object for rendering
    const handleReactionCreationFailure = (error) => {
      reactionRunResults.push({
        error: 'Failed to submit ' + reaction.pipeline + ' for ' + sha256 + ': ' + error,
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    };

    const res = await createReaction(reaction, handleReactionCreationFailure);
    if (res) {
      // return response including reaction uuid and pipeline/group
      reactionRunResults.push({
        id: res.id,
        error: '',
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    }
  }
  return reactionRunResults;
};

// delete reactions
const deleteReactions = async (reactionList) => {
  const reactionDeleteResults = [];
  for (const reaction of reactionList) {
    // handle adding the error to the results object for rendering
    const handleReactionDeleteFailure = (error) => {
      reactionDeleteResults.push({
        error: 'Failed to delete ' + reaction.pipeline + ': ' + error,
      });
    };

    const res = await deleteReaction(reaction.group, reaction.id, handleReactionDeleteFailure);
    if (res) {
      // return response including reaction uuid and pipeline/group
      reactionDeleteResults.push({
        id: reaction.id,
        error: '',
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    }
  }
  return reactionDeleteResults;
};

// Component for allowing a user to select pipelines to run on a given sha256
const SelectPipelines = ({ userInfo, setReactionsList, setError, currentSelections }) => {
  const [pipelines, setPipelines] = useState({});
  const [selectedPipelines, setSelectedPipelines] = useState({});
  const [pipelinesListErrors, setPipelinesListErrors] = useState([]);
  // Get detailed pipelines info and a list of groups names
  useEffect(() => {
    let isSubscribed = true;
    const fetchData = async () => {
      const allPipelines = [];
      // dictionaries of name: boolean pairs representing whether a user has
      // selected a given pipeline to run or group to give access to a set
      // of uploaded files
      const selectablePipelines = {};
      const errors = [];
      if (userInfo && userInfo.groups) {
        for (const group of userInfo.groups) {
          const groupPipelines = await listPipelines(group, (error) => errors.push(error), true);
          if (groupPipelines) {
            allPipelines[group] = [...groupPipelines];
            groupPipelines.map((pipeline) => {
              // selectablePipelines =
              //   {group1: {pipeline1: false, pipeline2: false}, group2: {pipeline3: false}}
              const tempSelected = {};
              tempSelected[`${pipeline.name}`] = false;
              if (pipeline.name in selectablePipelines) {
                selectablePipelines[`${pipeline.group}`][`${pipeline.name}`] = false;
              } else {
                selectablePipelines[`${pipeline.group}`] = tempSelected;
              }
            });
          }
        }
      }
      if (currentSelections) {
        currentSelections.forEach((pipeline) => (selectablePipelines[`${pipeline.pipeline}_${pipeline.group}`] = true));
      }
      // set local errors from listing pipelines in each group
      setPipelinesListErrors(errors);
      // don't update if resource is no longer mounted
      if (isSubscribed) {
        // save detailed pipeline info and map of pipeline names
        setPipelines(allPipelines);
        setSelectedPipelines(selectablePipelines);
      }
    };
    fetchData();
    return () => {
      isSubscribed = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userInfo]);

  return (
    <Card className="panel">
      <Card.Body className="py-0">
        <center>
          {pipelinesListErrors &&
            pipelinesListErrors.map((error, idx) => (
              <Alert key={`pipeline-list-error-${idx}`} variant="" className="d-flex justify-content-center danger">
                {error}
              </Alert>
            ))}
          {pipelines &&
            userInfo &&
            userInfo['groups'] &&
            userInfo.groups.map((group) => {
              if (pipelines[group] && pipelines[group].length) {
                return (
                  <div key={group}>
                    <Row className="mt-4 mb-2">
                      <Col>
                        <Title small>{group}</Title>
                      </Col>
                    </Row>
                    <Row className="mb-4">
                      <Col>
                        {pipelines[group]
                          .sort((a, b) => orderComparePipelineName(a, b))
                          .map((pipeline) =>
                            pipeline.description != null ? (
                              <OverlayTipTop
                                key={`${pipeline.group}_${pipeline.name}`}
                                wide
                                tip={<MarkdownHtml remarkPlugins={[remarkGfm]}>{pipeline.description}</MarkdownHtml>}
                              >
                                <Button
                                  variant=""
                                  // eslint-disable-next-line max-len
                                  className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`][`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                  onClick={(e) => {
                                    const selected = structuredClone(selectedPipelines);
                                    selected[`${pipeline.group}`][`${pipeline.name}`] =
                                      !selectedPipelines[`${pipeline.group}`][`${pipeline.name}`];
                                    setSelectedPipelines(selected);
                                    setReactionsList(buildReactionsList(selected));
                                    if (setError) setError([]);
                                  }}
                                >
                                  <font size="3">
                                    <b>{pipeline.name}</b>
                                  </font>
                                </Button>
                              </OverlayTipTop>
                            ) : (
                              <Button
                                variant=""
                                key={`${pipeline.group}_${pipeline.name}`}
                                // eslint-disable-next-line max-len
                                className={`m-1 primary-btn ${selectedPipelines[`${pipeline.group}`][`${pipeline.name}`] ? 'selected' : 'unselected'}`}
                                onClick={(e) => {
                                  const selected = structuredClone(selectedPipelines);
                                  selected[`${pipeline.group}`][`${pipeline.name}`] =
                                    !selectedPipelines[`${pipeline.group}`][`${pipeline.name}`];
                                  /* const selected = {...selectedPipelines,
                                [`${pipeline.name}_${pipeline.group}`]:
                                !selectedPipelines[`${pipeline.name}_${pipeline.group}`]};*/
                                  setSelectedPipelines(selected);
                                  setReactionsList(buildReactionsList(selected));
                                  if (setError) setError([]);
                                }}
                              >
                                <font size="3">
                                  <b>{pipeline.name}</b>
                                </font>
                              </Button>
                            ),
                          )}
                      </Col>
                    </Row>
                  </div>
                );
              } else {
                return null;
              }
            })}
        </center>
      </Card.Body>
    </Card>
  );
};

// Alert component for error and info responses for component submission
const RunReactionAlerts = ({ responses }) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((runResponse, idx) => (
          <div className="my-1" key={idx}>
            {runResponse.error && (
              <Alert className="full-width" variant="danger">
                <center>{runResponse.error}</center>
              </Alert>
            )}
            {runResponse.error == '' && (
              <Alert className="my-2 full-width" variant="info">
                <center>
                  <span>
                    {`Successfully submitted reaction `}
                    <Link className="link-text" to={`/reaction/${runResponse.group}/${runResponse.id}`} target="_blank">
                      {runResponse.id}
                    </Link>
                    {` for pipeline ${runResponse.pipeline} from group ${runResponse.group}!`}
                  </span>
                </center>
              </Alert>
            )}
          </div>
        ))}
    </>
  );
};

// Alert component for error and info responses for component deletion
const DeleteReactionAlerts = ({ responses }) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((deleteResponse, idx) => (
          <Row key={idx}>
            {deleteResponse.error && (
              <Alert className="full-width danger" variant="">
                <center>{deleteResponse.error}</center>
              </Alert>
            )}
            {deleteResponse.error == '' && (
              <Alert className="full-width info" variant="">
                <center>
                  <span>
                    {`Successfully deleted reaction ${deleteResponse.id}`}
                    {` for pipeline ${deleteResponse.pipeline} from group ${deleteResponse.group}!`}
                  </span>
                </center>
              </Alert>
            )}
          </Row>
        ))}
    </>
  );
};

// sort pipelines when displaying them in a list
const orderComparePipelineName = (a, b) => {
  return a.name.localeCompare(b.name);
};

// sort pipelines when displaying them in a list
const orderComparePipeline = (a, b) => {
  return (a.group + a.name).localeCompare(b.group + b.name);
};

const RunPipelines = ({ sha256 }) => {
  const { userInfo } = useAuth();
  const [reactionsList, setReactionsList] = useState([]);
  const [runReactionResponses, setRunReactionResponses] = useState([]);
  const [running, setRunning] = useState(false);

  // handle the reaction submission and setting of responses
  // this must be wrapped in a function object because of the async call
  const handleSubmitReactions = async () => {
    setRunning(true);
    const runResponses = await submitReactions(sha256, reactionsList);
    setRunReactionResponses(runResponses);
    setRunning(false);
  };

  return (
    <div id="runpipelines-tab">
      <SelectPipelines userInfo={userInfo} setReactionsList={setReactionsList} sha256={sha256} />
      <RunReactionAlerts responses={runReactionResponses} />
      <Row className="d-flex justify-content-center mt-2">
        {running ? (
          <LoadingSpinner loading={running}></LoadingSpinner>
        ) : (
          <Button className="ok-btn auto-width" onClick={() => handleSubmitReactions()}>
            Run Pipelines
          </Button>
        )}
      </Row>
    </div>
  );
};

const ReactionStatus = ({ sha256, autoRefresh }) => {
  const [loading, setLoading] = useState(false);
  const [reactionsList, setReactionsList] = useState([]);
  const [reactionsMap, setReactionsMap] = useState({});
  const [reactionsListSelections, setReactionsListSelections] = useState({});
  const [reactionsAllSelected, setReactionsAllSelected] = useState(false);
  const { userInfo, checkCookie } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showDeleteItems] = useState(5);
  const [deleteReactionResponses, setDeleteReactionResponses] = useState([]);
  // get a list of reactions by the sha256 tag
  useEffect(() => {
    const getReactionsList = async () => {
      if (!deleteInProgress) {
        const reactions = [];
        if (userInfo && userInfo.groups) {
          for (const group of userInfo.groups) {
            let moreReactions = true;
            let cursor = null;
            // need to get all reactions in chunks of 100 until there are no more left
            while (moreReactions) {
              const reactionsList = await listReactions(group, checkCookie, '', sha256, true, cursor, 10000);
              if (reactionsList) {
                // add returned reactions to local reactions array
                reactions.push(...reactionsList.details);
                // if cursor is undefined there are no more reactions for this group/tag
                if (reactionsList['cursor'] == undefined) {
                  moreReactions = false;
                } else {
                  cursor = reactionsRes.data.cursor;
                }
              }
            }
          }

          setReactionsList(reactions);
          reactions.forEach((reaction) => (reactionsMap[reaction.id] = reaction));
          setReactionsMap(reactionsMap);
        }
        setLoading(false);
      }
    };

    // only trigger reaction status API requests when component is being viewed
    if (autoRefresh) {
      // get a reaction list for the first render
      setLoading(true);
      getReactionsList();
      // now update the list every X seconds where x is the interval passed in below
      const intervalId = setInterval(() => {
        getReactionsList();
      }, 30000);
      return () => {
        clearInterval(intervalId);
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userInfo, sha256, autoRefresh]);

  const handleSelectionChange = (key) => {
    setDeleteReactionResponses([]);
    setReactionsListSelections((prevState) => {
      const newState = { ...prevState };
      if (prevState[key] == undefined) {
        newState[key] = true;
      } else {
        newState[key] = !prevState[key];
      }
      return newState;
    });
    setReactionsAllSelected(false);
  };

  const handleSelectAll = () => {
    setDeleteReactionResponses([]);
    const newSelections = {};
    for (const reaction in reactionsList) {
      if (reactionsList[reaction].id) {
        newSelections[reactionsList[reaction].id] = !reactionsAllSelected;
      }
    }
    setReactionsListSelections(newSelections);
    setReactionsAllSelected(!reactionsAllSelected);
  };

  const handleShowDeleteModal = () => {
    setShowDeleteModal(true);
  };

  const handleCloseDeleteModal = () => {
    setShowDeleteModal(false);
  };

  const anySelected = () => {
    // Check is any of the selection items are actually selected.
    const values = Object.values(reactionsListSelections);
    for (let i = 0; i < values.length; i++) {
      if (values[i]) {
        return true;
      }
    }
    return false;
  };

  const truncateSelections = (selections) => {
    // Function for displaying truncated list of reactions that will be deleted
    let truncatedList = [];
    Object.keys(selections).map((reactionID, idx) => {
      if (selections[reactionID]) {
        truncatedList.push(`${reactionsMap[reactionID].pipeline} : 
          ${reactionsMap[reactionID].group}`);
      }
    });
    if (truncatedList.length > 5) {
      const truncateMsg = `${truncatedList.length - 5} more selections ...`;
      truncatedList = truncatedList.slice(0, showDeleteItems);
      truncatedList.push(truncateMsg);
    }
    return truncatedList;
  };

  const handleDeleteClick = async () => {
    setShowDeleteModal(false);
    setLoading(true);
    deleteInProgress = true;
    setReactionsListSelections({});
    setReactionsAllSelected(false);

    const deleteReactionList = [];
    Object.keys(reactionsListSelections).map(async (reactionID, idx) => {
      if (reactionsListSelections[reactionID]) {
        deleteReactionList.push(reactionsMap[reactionID]);
      }
    });
    const deleteResponses = await deleteReactions(deleteReactionList);
    setDeleteReactionResponses(deleteResponses);

    setLoading(false);
    deleteInProgress = false;
  };

  return (
    <div id="reactionstatus-tab" className="mx-4">
      {!loading && reactionsList.length == 0 ? (
        <>
          <Alert variant="" className="info">
            <Alert.Heading>
              <center>
                <h3>No Reactions Found</h3>
              </center>
            </Alert.Heading>
            <center>
              <p>Create a reaction and then check the status here</p>
            </center>
          </Alert>
        </>
      ) : (
        <>
          <LoadingSpinner loading={loading}></LoadingSpinner>
          {loading ? (
            <></>
          ) : (
            <>
              <Row>
                <Card className="panel">
                  <Row>
                    <Col className="reactions-pipeline mt-3" md={2}>
                      <Subtitle>Pipeline</Subtitle>
                    </Col>
                    <Col className="reactions-creator mt-3" md={1}>
                      <Subtitle className="mt-2">Creator</Subtitle>
                    </Col>
                    <Col className="reactions-group mt-3" md={1}>
                      <Subtitle>Group</Subtitle>
                    </Col>
                    <Col className="reactions-status mt-3" md={1}>
                      <Subtitle>Status</Subtitle>
                    </Col>
                    <Col className="reactions-id mt-3" md={3}>
                      <Subtitle>Reaction ID</Subtitle>
                    </Col>
                    <Col className="reactions-selection d-flex justify-content-end" md={1}>
                      <ButtonToolbar className="d-flex justify-content-end">
                        <OverlayTipTop
                          tip={`Delete selected reactions. Only system admins,
                          group owners/managers, and the submitter can delete a reaction.`}
                        >
                          <Button
                            size="sm"
                            className="icon-btn me-2 my-1"
                            variant=""
                            disabled={!anySelected()}
                            onClick={handleShowDeleteModal}
                          >
                            <FaTrash />
                          </Button>
                        </OverlayTipTop>
                        <OverlayTipLeft tip={'Select All Reactions'}>
                          <FormCheck onChange={handleSelectAll} className="mt-2" checked={reactionsAllSelected}></FormCheck>
                        </OverlayTipLeft>
                      </ButtonToolbar>
                      <Modal show={showDeleteModal} onHide={handleCloseDeleteModal} backdrop="static" keyboard={false}>
                        <Modal.Header closeButton>
                          <Modal.Title>Confirm deletion?</Modal.Title>
                        </Modal.Header>
                        <Modal.Body>
                          <p>Do you really want to delete the following reactions:</p>
                          {truncateSelections(reactionsListSelections).map((reactionString, idx) => {
                            return (
                              <div key={idx}>
                                <center>
                                  <b>{reactionString}</b>
                                </center>
                              </div>
                            );
                          })}
                          <center>
                            <p>
                              <b>{}</b>
                              <b>{}</b>
                            </p>
                          </center>
                        </Modal.Body>
                        <Modal.Footer className="d-flex justify-content-center">
                          <Button className="danger-btn" onClick={handleDeleteClick}>
                            Confirm
                          </Button>
                          <Button className="primary-btn" onClick={handleCloseDeleteModal}>
                            Cancel
                          </Button>
                        </Modal.Footer>
                      </Modal>
                    </Col>
                  </Row>
                </Card>
              </Row>
              <Row className="mt-1">
                {reactionsList.map((reaction, idx) => (
                  <Card key={`${reaction.id}_${idx}`} className="highlight-card">
                    <Row>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-pipeline" md={2}>
                        <Col>{reaction.pipeline}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-creator" md={1}>
                        <Col>{reaction.creator}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-group" md={1}>
                        <Col>{reaction.group}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-status" md={1}>
                        <Col>{getStatusBadge(reaction.status)}</Col>
                      </Link>
                      <Link to={`/reaction/${reaction.group}/${reaction.id}`} className="no-decoration reactions-id" md={3}>
                        <Col>{reaction.id}</Col>
                      </Link>
                      <Col className="reactions-selection" md={1}>
                        <FormCheck
                          onChange={() => {
                            handleSelectionChange(reaction.id);
                          }}
                          checked={reactionsListSelections[reaction.id] ? reactionsListSelections[reaction.id] : false}
                        ></FormCheck>
                      </Col>
                    </Row>
                  </Card>
                ))}
              </Row>
            </>
          )}
        </>
      )}
      <Row className="pt-1">
        <DeleteReactionAlerts responses={deleteReactionResponses} />
      </Row>
      <br />
      <br />
    </div>
  );
};

export {
  SelectPipelines,
  RunPipelines,
  RunReactionAlerts,
  submitReactions,
  deleteReaction,
  ReactionStatus,
  getStatusBadge,
  getStatusIcon,
  orderComparePipeline,
};
